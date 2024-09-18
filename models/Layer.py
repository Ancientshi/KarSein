import torch
import torch.nn.functional as F
import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


seed= 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class KANLinear2D(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        emb_dim=64,
    ):
        super(KANLinear2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.emb_dim=emb_dim

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.B_spline_weight = torch.nn.Parameter(
            torch.Tensor(in_features,grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )


        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        
            
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, 1)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            coeff=self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                ).squeeze(0)
            scale_spline=(self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
            self.B_spline_weight.data.copy_(
                coeff
                * 
                scale_spline
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
                
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, 1)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            1,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
             
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler
            if self.enable_standalone_scale_spline
            else 1.0
        )
              
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        batch_size, emb_dim, d = x.shape
        original_shape = x.shape

        base_activated_x = self.base_activation(x)  # (batch, emb_dim, in_features)
        base_output = torch.matmul(base_activated_x, self.base_weight.t())
        
        vmap_b_splines=torch.vmap(self.b_splines, in_dims=1, out_dims=1)
        # (batch, emb_dim, in_features, grid_size + spline_order) 64 6 13 
        basis_activated_x = vmap_b_splines(x)  
        
        b_spline_activated_x = torch.einsum('jk, bijk -> bij', self.B_spline_weight, basis_activated_x)
        spline_output = torch.matmul(b_spline_activated_x, self.scaled_spline_weight.t())
        
        output = base_output + spline_output
        return output  
        

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        regularization_loss_activation = self.spline_weight.abs().sum()
        p = self.spline_weight.abs() / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * (p+ 1e-4).log())

        spline_weight_reg=regularize_activation * regularization_loss_activation+ regularize_entropy * regularization_loss_entropy
            
        reg_result=0.1*spline_weight_reg
        
        # regularization_loss_activation = self.base_weight.abs().sum()
        # p = self.base_weight.abs() / regularization_loss_activation
        # regularization_loss_entropy = -torch.sum(p * (p+ 1e-4).log())
        # base_weight_reg=regularize_activation * regularization_loss_activation+ regularize_entropy * regularization_loss_entropy
        
        # reg_result+=0.001*base_weight_reg
        
        return (
            reg_result
        )
        
        
    def get_posact(self, x):
        assert x.size(-1) == self.in_features
        batch_size, emb_dim, d = x.shape
        original_shape = x.shape

        base_activation_output = self.base_activation(x)  # (batch, emb_dim, in_features)
        base_activation_output_expanded = base_activation_output.unsqueeze(-1)  # (batch, emb_dim, in_features, 1)
        base_weight_expanded = self.base_weight.unsqueeze(0)  # (1, out_features, in_features)
        base_output = base_activation_output_expanded * base_weight_expanded.permute(0, 2, 1)  # (batch, emb_dim, in_features, out_features)

        
        vmap_b_splines=torch.vmap(self.b_splines, in_dims=1, out_dims=1)
        # (batch, emb_dim, in_features, grid_size + spline_order) 64 6 13 
        basis_activated_x = vmap_b_splines(x)  
        
        b_spline_activated_x = torch.einsum('jk, bijk -> bij', self.B_spline_weight, basis_activated_x)
        b_spline_activated_x_expanded = b_spline_activated_x.unsqueeze(-1)  # (batch, emb_dim, in_features, 1)
        scaled_spline_weight_expanded = self.scaled_spline_weight.unsqueeze(0) # (1, out_features, in_features)
        spline_output= b_spline_activated_x_expanded * scaled_spline_weight_expanded.permute(0, 2, 1)  # (batch, emb_dim, in_features, out_features)
        
        output = base_output + spline_output
        return output,base_activation_output+b_spline_activated_x

        
        
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features )
        )
        self.B_spline_weight = torch.nn.Parameter(
            torch.Tensor(in_features, grid_size + spline_order)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, 1)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            coeff=self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                ).squeeze(0)
            scale_spline=(self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
            
            self.B_spline_weight.data.copy_(
                coeff
                * 
                scale_spline
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, 1)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            1,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler
            if self.enable_standalone_scale_spline
            else 1.0
        )
    

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features

        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        basis_activated_x = self.b_splines(x) 
        b_spline_activated_x = torch.einsum('bik,ik->bi', basis_activated_x, self.B_spline_weight)  
        spline_output = F.linear(b_spline_activated_x, self.scaled_spline_weight)
        output = base_output + spline_output
        return output


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        regularization_loss_activation = self.spline_weight.abs().sum()
        p = self.spline_weight.abs() / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * (p+ 1e-4).log())

        spline_weight_reg=regularize_activation * regularization_loss_activation+ regularize_entropy * regularization_loss_entropy
            
        reg_result=0.1*spline_weight_reg
        
        # regularization_loss_activation = self.base_weight.abs().sum()
        # p = self.base_weight.abs() / regularization_loss_activation
        # regularization_loss_entropy = -torch.sum(p * (p+ 1e-4).log())
        # base_weight_reg=regularize_activation * regularization_loss_activation+ regularize_entropy * regularization_loss_entropy
        
        # reg_result+=0.001*base_weight_reg
        
        return (
            reg_result
        )
    
        
    def get_posact(self, x):
        assert x.size(-1) == self.in_features

        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_activation_output = self.base_activation(x)  # (batch, in_features)
        # Expand dimensions for element-wise multiplication
        base_activation_output_expanded = base_activation_output.unsqueeze(-1)  # (batch, in_features, 1)
        base_weight_expanded = self.base_weight.unsqueeze(0)  # (1, out_features, in_features)
        # Perform element-wise multiplication
        base_output = base_activation_output_expanded * base_weight_expanded.permute(0, 2, 1)  # (batch, in_features, out_features)
    
    
        # Compute spline output similarly
        basis_activated_x = self.b_splines(x) 
        b_spline_activated_x = torch.einsum('bik,ik->bi', basis_activated_x, self.B_spline_weight)  
        b_spline_activated_x_expanded = b_spline_activated_x.unsqueeze(-1)  # (batch, in_features, 1)
        scaled_spline_weight_expanded = self.scaled_spline_weight.unsqueeze(0)  # (1, out_features, in_features)
        spline_output = b_spline_activated_x_expanded * scaled_spline_weight_expanded.permute(0, 2, 1)  # (batch, in_features, out_features)

        output = base_output + spline_output
        return output,base_activation_output+b_spline_activated_x

    
    
class KarSein_Layer(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        emb_dim=-1,
    ):
        super(KarSein_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()

        if emb_dim==-1:
            for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
                self.layers.append(
                    KANLinear(
                        in_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
                )
        else:
            for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
                self.layers.append(
                    KANLinear2D(
                        in_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                        emb_dim=emb_dim
                    )
                )


    def forward(self, x: torch.Tensor, update_grid=False):
        if hasattr(self, 'block_index'):
            if len(x.shape) == 2:
                for layer_index, layer in enumerate(self.layers):
                    if update_grid:
                        layer.update_grid(x)
                    x[:,self.block_index[layer_index]] = 0
                    x = layer(x)
                return x
            elif len(x.shape) == 3:
                for layer_index, layer in enumerate(self.layers):
                    if update_grid:
                        layer.update_grid(x)
                    x[:,:,self.block_index[layer_index]] = 0
                    x = layer(x)
                return x
        else:
            for layer in self.layers:
                if update_grid:
                    layer.update_grid(x)
                x = layer(x)
            return x

    
    def draw_activation(self, folder, layer_index, x, activated_x):
        folder = f'{folder}/activation/layer_{layer_index}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if len(x.shape) == 2:
            batch_size, d = x.shape
            for d_index in range(d):
                input = x[:, d_index].cpu().detach().numpy()
                output = activated_x[:, d_index].cpu().detach().numpy()
                
                # Sort input and output by the sorted input values
                index = np.argsort(input)
                input = input[index]
                output = output[index]
                
                # Fit the data using a 3rd degree polynomial: w1*x^3 + w2*x^2 + w3*x + c
                try:
                    coeffs = np.polyfit(input, output, 3)
                except:
                    coeffs = np.zeros(4)
                    
                poly = np.poly1d(coeffs)
                fitted_output = poly(input)
                
                # Calculate the RMSE
                rmse = np.sqrt(mean_squared_error(output, fitted_output))
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(input, output, color='black', linewidth=8, label='Original Data')
                ax.plot(input, fitted_output, color='red', linestyle='--', linewidth=2, label='Fitted Curve')
                
                # Remove axis ticks and labels, but keep the spines (borders)
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                
                # Set the color and linewidth for the spines
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(5)
                
                # Set the title for the plot with the formula and RMSE
                formula = f"$y = {coeffs[0]:.4f}x^3 + {coeffs[1]:.4f}x^2 + {coeffs[2]:.4f}x + {coeffs[3]:.4f}$\nRMSE = {rmse:.4f}"
                ax.set_title(formula)
                
                # Save the figure
                plt.savefig(f"{folder}/feature_{d_index}.png", bbox_inches='tight')
                plt.close(fig)
        
        elif len(x.shape) == 3:
            batch_size, emb_dim, d = x.shape
            for emb_index in range(emb_dim):
                for d_index in range(d):
                    input = x[:, emb_index, d_index].cpu().detach().numpy()
                    output = activated_x[:, emb_index, d_index].cpu().detach().numpy()
                    
                    # Sort input and output by the sorted input values
                    index = np.argsort(input)
                    input = input[index]
                    output = output[index]
                    
                    # Fit the data using a 3rd degree polynomial: w1*x^3 + w2*x^2 + w3*x + c
                    try:
                        coeffs = np.polyfit(input, output, 3)
                    except:
                        coeffs = np.zeros(4)
                    poly = np.poly1d(coeffs)
                    fitted_output = poly(input)
                    
                    # Calculate the RMSE
                    rmse = np.sqrt(mean_squared_error(output, fitted_output))
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.plot(input, output, color='black', linewidth=8, label='Original Data')
                    ax.plot(input, fitted_output, color='red', linestyle='--', linewidth=2, label='Fitted Curve')
                    
                    # Remove axis ticks and labels, but keep the spines (borders)
                    ax.set_xticks([])
                    
                    # Set the color and linewidth for the spines
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(5)
                    
                    # Set the title for the plot with the formula and RMSE
                    formula = f"$y = {coeffs[0]:.4f}x^3 + {coeffs[1]:.4f}x^2 + {coeffs[2]:.4f}x + {coeffs[3]:.4f}$\nRMSE = {rmse:.4f}"
                    ax.set_title(formula)
                     
                    # Save the figure
                    plt.savefig(f"{folder}/feature_{d_index}_emb{emb_index}.png", bbox_inches='tight')
                    plt.close(fig)
                    
        
    def get_posact(self,folder,x):
        self.pos_act=[] #batch,in,out
        for layer_index,layer in enumerate(self.layers):
            pos_act,activated_x = layer.get_posact(x)
            if folder is not None:
                self.draw_activation(folder,layer_index,x,activated_x)
            x = layer(x)
            self.pos_act.append(pos_act)
    
    def plot(self,folder,module_name,x):
        self.get_posact(folder,x)
        '''
        Visualize the overall network structure. Each layer's transformation from input features to output features is represented by a matrix.
        The intensity of the matrix elements indicates the weights of the spline scaler.
        '''
        
        if len(x.shape) == 2:
            for layer_index, layer in enumerate(self.layers):
                fig, ax = plt.subplots(figsize=(16, 20))
                mean=torch.mean(self.pos_act[layer_index],dim=0).cpu().detach().numpy()
                
                # Apply the threshold condition
                mean_abs = np.abs(mean)
                
                threshold=0.01
                index = np.where(np.all(mean_abs <= threshold, axis=1))[0]
                prune_ratio = len(index) / mean.shape[0]
                
                mean[mean_abs <= 0.01] = 0
                mean=np.abs(mean)*20
                mean=np.tanh(mean)
            
                cax = ax.matshow(mean, cmap="Greys")
                fig.colorbar(cax)

                ax.set_title(f"Layer {layer_index}, prune ratio: {prune_ratio}", pad=20)
                ax.set_xlabel("Output Features")
                ax.set_ylabel("Input Features")
                plt.grid(False)

                if not os.path.exists(folder):
                    os.makedirs(folder)
                plt.savefig(f"{folder}/{module_name}_layer_{layer_index}.png")
                plt.close(fig)  # Close the figure to avoid display in non-interactive environments
                
        elif len(x.shape) == 3:
            for emb_index in range(x.shape[1]):
                for layer_index, layer in enumerate(self.layers):
                    fig, ax = plt.subplots(figsize=(16, 20))
                    #(batch, emb_dim, in_features, out_features)
                    mean=torch.mean(self.pos_act[layer_index][:,emb_index,:,:],dim=0).cpu().detach().numpy()
                    
                    # Apply the threshold condition
                    mean_abs = np.abs(mean)
                    
                    threshold=0.01
                    index = np.where(np.all(mean_abs <= threshold, axis=1))[0]
                    prune_ratio = len(index) / mean.shape[0]
                    
                    mean[mean_abs <= 0.01] = 0
                    mean=np.abs(mean)*20
                    mean=np.tanh(mean)
                
                    cax = ax.matshow(mean, cmap="Greys")
                    fig.colorbar(cax)

                    ax.set_title(f"Layer {layer_index}, prune ratio: {prune_ratio}", pad=20)
                    ax.set_xlabel("Output Features")
                    ax.set_ylabel("Input Features")
                    
                    plt.grid(False)

                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(f"{folder}/{module_name}_layer_{layer_index}_emb{emb_index}.png")
                    plt.close(fig)
    
    
    def prune(self, x, threshold=0.01):
        self.get_posact(None,x)
        self.block_index = []
        if len(x.shape) == 2:
            for layer_index, layer in enumerate(self.layers):
                mean = torch.mean(self.pos_act[layer_index], dim=0).cpu().detach().numpy()
                mean_abs = np.abs(mean)
                index = np.where(np.all(mean_abs <= threshold, axis=1))[0]

                with torch.no_grad():
                    layer.base_weight[:, index] = 1e-10
                    layer.spline_weight[:, index] = 1e-10
                
                layer.base_weight[:, index].requires_grad = False
                layer.spline_weight[:, index].requires_grad = False
                
                prune_ratio = len(index) / mean.shape[0]
                print(f'layer {layer_index} prune ratio: {prune_ratio}')
                
                self.block_index.append(index)
        elif len(x.shape) == 3:
            for layer_index, layer in enumerate(self.layers):
                index_set=[]
                for emb_index in range(x.shape[1]):
                    for layer_index, layer in enumerate(self.layers):
                        mean = torch.mean(self.pos_act[layer_index][:,emb_index,:,:], dim=0).cpu().detach().numpy()
                        mean_abs = np.abs(mean)
                        index = np.where(np.all(mean_abs <= threshold, axis=1))[0]
                        index_list=index.tolist()
                        index_set.append(set(index_list))

                intersaction_index = list(set.intersection(*index_set))
                prune_ratio = len(intersaction_index) / mean.shape[0]
                print(f'layer {layer_index} prune ratio: {prune_ratio}')
                self.block_index.append(intersaction_index)

        
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )