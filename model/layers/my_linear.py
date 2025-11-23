import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.quantisation.observer import Observer, FakeQuantize, quantize_tensor, dequantize_tensor

class MyLinear(nn.Module):
    '''Quantized version of Linear layer.'''
    def __init__(self, 
                 input_dim: int = 1, 
                 output_dim: int = 4,
                 bias:bool = True,
                 num_bits:int = 8):
        super().__init__()
        
        '''Initialize standard layers.'''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.num_bits = num_bits
        
        self.num_bits_obs = 32

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        self.reset_parameters()

        '''Modes for calibration and quantization'''
        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_input = Observer(num_bits=num_bits)
        self.observer_weight = Observer(num_bits=num_bits)
        self.observer_output = Observer(num_bits=num_bits)

        self.register_buffer('m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_in', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('num_bits_model', torch.tensor(num_bits, requires_grad=False))
        self.register_buffer('num_bits_scale', torch.tensor(self.num_bits_obs, requires_grad=False))

        self.use_obs = False
        self.first_layer = False

    def reset_parameters(self):
        '''
            Reset parameters of the model
        '''
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Custom message function for PointNetConv.
            We select the message function based on the current mode (calibration, quantize, or float)
        '''
        if self.calib_mode.item() and not self.quantize_mode.item():
            return self.forward_calib(x)
        elif self.quantize_mode.item():
            return self.forward_quant(x)
        elif not self.calib_mode.item() and not self.quantize_mode.item():
            return self.forward_float(x)
        else:
            raise ValueError('Invalid mode')

    def forward_float(self, 
                x: torch.Tensor):

        '''Standard forward pass of Linear layer.'''
        return self.linear(x)
    
    def forward_calib(self, 
                    x: torch.Tensor):
        
        '''Calibration forward for updating observers.'''
        if self.use_obs:
            '''Update input observer.'''
            if self.training:
                self.observer_input.update(x)
            x = FakeQuantize.apply(x, self.observer_input)

        '''Update weight observer and propagate message through linear layer.'''
        if self.training:
            self.observer_weight.update(self.linear.weight.data)

        if self.bias:
            x = F.linear(x, FakeQuantize.apply(self.linear.weight, self.observer_weight), self.linear.bias)
        else:
            x = F.linear(x, FakeQuantize.apply(self.linear.weight, self.observer_weight))
        
        '''Update output observer and calculate output.'''
        if self.training:
            self.observer_output.update(x)
        x = FakeQuantize.apply(x, self.observer_output)
        return x

    def forward_quant(self, 
                  x: torch.Tensor):
        
        '''Quantized forward pass of Linear layer.'''
        
        '''Quantize input x'''
        if self.first_layer:
            '''We need to quantize x.'''
            x = self.observer_input.quantize_tensor(x)
            x = x - self.observer_input.zero_point
        else:
            '''For other layers, we do not need to quantize x'''
            x = x - self.observer_input.zero_point
        x = self.linear(x)
        x = (x * self.m).round() + self.observer_output.zero_point
        x = torch.clamp(x, 0, 2**self.num_bits - 1)
        return x

    def calibrate(self):
        self.calib_mode.fill_(True)

    def quantize(self,
               observer_input: Observer = None,
               observer_output: Observer = None):
        '''
            Quantize model - quantize weights/bias and calculate scales
        '''
        self.quantize_mode.fill_(True)

        if observer_input is not None:
            self.observer_input = observer_input
        if observer_output is not None:
            self.observer_output = observer_output

        # Quantize scales for input, weight, and output
        self.qscale_in.copy_( (2**self.num_bits_obs * self.observer_input.scale).round() )
        self.observer_input.scale = self.qscale_in / (2 ** self.num_bits_obs)

        self.qscale_w.copy_( (2**self.num_bits_obs * self.observer_weight.scale).round() )
        self.observer_weight.scale = self.qscale_w / (2 ** self.num_bits_obs)

        self.qscale_out.copy_( (2**self.num_bits_obs * self.observer_output.scale).round() )
        self.observer_output.scale = self.qscale_out / (2 ** self.num_bits_obs)

        # Compute scaling factor m
        qscale_m = (self.observer_weight.scale * self.observer_input.scale) / self.observer_output.scale
        self.qscale_m.copy_( (2**self.num_bits_obs * qscale_m).round() )
        self.m.copy_(self.qscale_m / (2 ** self.num_bits_obs))
            
        self.linear.weight = torch.nn.Parameter(self.observer_weight.quantize_tensor(self.linear.weight))
        self.linear.weight = torch.nn.Parameter(self.linear.weight - self.observer_weight.zero_point)

        if self.bias:
            self.linear.bias = torch.nn.Parameter(quantize_tensor(self.linear.bias, 
                                        scale=self.observer_input.scale * self.observer_weight.scale,
                                        zero_point=0, 
                                        num_bits=32, 
                                        signed=True))

    def get_parameters(self,
                       file_name: str = None):
        
        with open(file_name, 'w') as f:
            '''Save scales and zero points to file.'''
            f.write(f"Input scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_in)}\n")
            f.write(f"Input zero point:\n {int(self.observer_input.zero_point)}\n")
            f.write(f"Weight scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_w)}\n")
            f.write(f"Weight zero point:\n {int(self.observer_weight.zero_point)}\n")
            f.write(f"Output scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_out)}\n")
            f.write(f"Output zero point:\n {int(self.observer_output.zero_point)}\n")
            f.write(f"M Scales ({int(self.num_bits_obs)} bit):\n {int(self.qscale_m)}\n")

            '''Save weights and bias to file.'''
            weight = torch.flip(self.linear.weight, [1])
            weight = weight.detach().cpu().numpy().astype(np.int32).tolist()
            
            f.write(f"Weight ({int(self.num_bits)} bit):\n")
            for idx, w in enumerate(weight):
                f.write(f"weights_conv[{idx}] = {str(w).replace('[', '{').replace(']', '}') + ';'}\n")

            if self.bias:
                bias = torch.flip(self.linear.bias, [0])
                bias = bias.detach().cpu().numpy().astype(np.int32).tolist()
                f.write(f"\nBias ({int(self.num_bits)} bit):\n {str(bias).replace('[', '{').replace(']', '}') + ';'}\n")

            with open(file_name.replace('.txt', '.mem'), 'w') as f:
                for idx, we in enumerate(weight):
                    bin_vec = [np.binary_repr(w+self.observer_weight.zero_point.to(torch.int32).item(), width=9)[1:] for w in we]
                    Z1a2 = sum([w + self.observer_weight.zero_point.to(torch.int32).item() for w in we]) * \
                        self.observer_input.zero_point.to(torch.int32).item()
                    NZ1Z2 = self.input_dim * \
                                self.observer_input.zero_point.to(torch.int32).item() * \
                                    self.observer_weight.zero_point.to(torch.int32).item()
                    # Concat to bin_vec binary repr of bias
                    bin_vec = bin_vec + [np.binary_repr(bias[len(bias)-idx-1] - Z1a2 + NZ1Z2, width=32)]
                    dlugi_ciag_bitow = ''.join(bin_vec)
                    wartosc_hex = hex(int(dlugi_ciag_bitow, 2))
                    f.write(f"{str(wartosc_hex)[2:]}\n")

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias}, num_bits={self.num_bits})"