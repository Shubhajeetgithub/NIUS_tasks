import numpy as np
from typing import Callable, Dict, Any, Optional, Union
class LeastSquareRegression:
    def __init__(self):
        self.results = {}
        self.fitted = False
    
    def fit(self, x: np.ndarray, y: np.ndarray, f: Callable, g: Callable, dfdx: Callable, dgdy: Callable, sigma_x: float, sigma_y: float) -> Dict[str, Any]:
        X = f(x)
        Y = g(y)
        sigma_X = np.abs(dfdx(x)) * sigma_x
        sigma_Y = np.abs(dgdy(y)) * sigma_y
        weights = 1 / (sigma_Y**2)
        S = np.sum(weights)
        S_X = np.sum(weights * X)
        S_Y = np.sum(weights * Y)
        S_XX = np.sum(weights * X * X)
        S_XY = np.sum(weights * X * Y)

        Delta = S * S_XX - S_X * S_X
        a = (S * S_XY - S_X * S_Y) / Delta
        b = (S_XX * S_Y - S_X * S_XY) / Delta
        sigma_a = np.sqrt(abs(S / Delta))
        sigma_b = np.sqrt(abs(S_XX / Delta))
        self.results = {
            'a': a,
            'b': b,
            'sigma_a': sigma_a,
            'sigma_b': sigma_b,
        }
        self.fitted = True
        return self.results

    def print_results(self) -> None:
        if not self.fitted:
            print("No fit has been performed yet.")
            return
        r = self.results
        print(f"a = {r['a']:.6f} ± {r['sigma_a']:.6f}")
        print(f"b = {r['b']:.6f} ± {r['sigma_b']:.6f}")


class MultidimensionalLeastSquareRegression:
    def __init__(self):
        self.results = {}
        self.fitted = False
        self.n_features = 0
        
    def fit(self, 
            x: np.ndarray, 
            y: np.ndarray, 
            f: Optional[Union[Callable, list]] = None, 
            g: Optional[Callable] = None,
            dfdx: Optional[Union[Callable, list]] = None, 
            dgdy: Optional[Callable] = None,
            sigma_x: Optional[Union[float, np.ndarray]] = None, 
            sigma_y: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit multidimensional least squares regression: y = A*x1 + B*x2 + C
        
        Parameters:
        -----------
        x : np.ndarray
            Input features of shape (n_samples, n_features) or (n_samples,) for 1D
        y : np.ndarray
            Target values of shape (n_samples,)
        f : Callable or list of Callables, optional
            Transformation function(s) for x. If None, identity function is used.
            If list, should have one function per feature.
        g : Callable, optional
            Transformation function for y. If None, identity function is used.
        dfdx : Callable or list of Callables, optional
            Derivatives of f with respect to x. If None, assumes derivative = 1.
        dgdy : Callable, optional
            Derivative of g with respect to y. If None, assumes derivative = 1.
        sigma_x : float or np.ndarray, optional
            Uncertainties in x. If None, assumes sigma_x = 1.
        sigma_y : float, optional
            Uncertainties in y. If None, assumes sigma_y = 1.
        """
        
        # Convert inputs to numpy arrays
        x = np.atleast_2d(x)
        if x.shape[0] == 1 and x.shape[1] > 1:
            x = x.T  # Transpose if we have (1, n_features) instead of (n_samples, 1)
        
        y = np.atleast_1d(y)
        n_samples, n_features = x.shape
        self.n_features = n_features
        
        # Set default transformation functions
        if f is None:
            f = [lambda xi: xi for _ in range(n_features)]
        elif not isinstance(f, list):
            f = [f for _ in range(n_features)]
            
        if g is None:
            g = lambda yi: yi
            
        if dfdx is None:
            dfdx = [lambda xi: np.ones_like(xi) for _ in range(n_features)]
        elif not isinstance(dfdx, list):
            dfdx = [dfdx for _ in range(n_features)]
            
        if dgdy is None:
            dgdy = lambda yi: np.ones_like(yi)
            
        # Set default uncertainties
        if sigma_x is None:
            sigma_x = np.ones((n_samples, n_features))
        elif np.isscalar(sigma_x):
            sigma_x = np.full((n_samples, n_features), sigma_x)
        else:
            sigma_x = np.atleast_2d(sigma_x)
            if sigma_x.shape[0] == 1:
                sigma_x = sigma_x.T
                
        if sigma_y is None:
            sigma_y = 1.0
            
        # Apply transformations
        X = np.zeros((n_samples, n_features))
        sigma_X = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            X[:, i] = f[i](x[:, i])
            sigma_X[:, i] = np.abs(dfdx[i](x[:, i])) * sigma_x[:, i]
            
        Y = g(y)
        sigma_Y = np.abs(dgdy(y)) * sigma_y
        
        # Build design matrix (add column of ones for intercept)
        A = np.column_stack([X, np.ones(n_samples)])
        
        # Calculate weights
        weights = 1 / (sigma_Y**2)
        W = np.diag(weights)
        
        # Weighted least squares solution: (A^T W A)^-1 A^T W Y
        AtWA = A.T @ W @ A
        AtWY = A.T @ W @ Y
        
        # Solve for coefficients
        coeffs = np.linalg.solve(AtWA, AtWY)
        
        # Calculate parameter uncertainties
        covariance_matrix = np.linalg.inv(AtWA)
        param_uncertainties = np.sqrt(np.diag(covariance_matrix))
        
        # Store results
        self.results = {
            'coefficients': coeffs[:-1],  # All but intercept
            'intercept': coeffs[-1],      # Last coefficient is intercept
            'covariance_matrix': covariance_matrix,
            'param_uncertainties': param_uncertainties,
            'n_features': n_features
        }
        
        # Store individual coefficient names and uncertainties
        coeff_names = [f'coeff_{i}' for i in range(n_features)] + ['intercept']
        for i, name in enumerate(coeff_names):
            self.results[name] = coeffs[i]
            self.results[f'sigma_{name}'] = param_uncertainties[i]
            
        self.fitted = True
        return self.results
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        x : np.ndarray
            Input features of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions.")
            
        x = np.atleast_2d(x)
        if x.shape[0] == 1 and x.shape[1] > 1:
            x = x.T
            
        predictions = np.dot(x, self.results['coefficients']) + self.results['intercept']
        return predictions
    
    def print_results(self) -> None:
        """Print the fitted parameters and their uncertainties."""
        if not self.fitted:
            print("No fit has been performed yet.")
            return
            
        r = self.results
        
        print("Fitted Model: y = ", end="")
        for i in range(self.n_features):
            coeff = r['coefficients'][i]
            sigma_coeff = r['param_uncertainties'][i]
            if i == 0:
                print(f"({coeff:.6f} ± {sigma_coeff:.6f})*x{i+1}", end="")
            else:
                sign = "+" if coeff >= 0 else ""
                print(f" {sign} ({coeff:.6f} ± {sigma_coeff:.6f})*x{i+1}", end="")
        
        intercept = r['intercept']
        sigma_intercept = r['param_uncertainties'][-1]
        sign = "+" if intercept >= 0 else ""
        print(f" {sign} ({intercept:.6f} ± {sigma_intercept:.6f})")
        
        print("\nIndividual Parameters:")
        for i in range(self.n_features):
            coeff = r['coefficients'][i]
            sigma_coeff = r['param_uncertainties'][i]
            print(f"Coefficient x{i+1}: {coeff:.6f} ± {sigma_coeff:.6f}")
        print(f"Intercept: {intercept:.6f} ± {sigma_intercept:.6f}")
        
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get the correlation matrix of the fitted parameters.
        
        Returns:
        --------
        np.ndarray
            Correlation matrix
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing correlation matrix.")
            
        cov = self.results['covariance_matrix']
        std_devs = np.sqrt(np.diag(cov))
        correlation_matrix = cov / np.outer(std_devs, std_devs)
        return correlation_matrix