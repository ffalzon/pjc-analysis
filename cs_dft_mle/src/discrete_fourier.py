import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.recovery_template import RecoveryTemplate

class DiscreteFourier(RecoveryTemplate):

    def setup(self, intersection_size: int):
        self.N = self.params.num_targets
        # self.s = intersection_size
        print(f"intersection_size: {intersection_size}")
        self.s = intersection_size

        A = np.zeros((2 * self.s, self.N), dtype=complex)

        # Construct the matrix A
        for k in range(2 * self.s):
            A[k, :] = np.exp(-2j * np.pi * k * np.arange(self.N) / self.N)
        self.A = A

    def get_A(self):
        return self.A

    def recover(self):
        # Solve the linear system to find p_hat
        p_hat = self.solve_for_p_hat(self.output_l, self.s)

        # Identify the support from the roots of the polynomial
        support_reconstructed = self.identify_support(p_hat, self.N)

        # Reconstruct the values at the identified support using the overdetermined system
        x_reconstructed = self.reconstruct_values_at_support(self.output_l, support_reconstructed, self.N)
        
        return x_reconstructed

    def set_inner_products(self, output_l):
        self.output_l = output_l

    def solve_for_p_hat(self, fourier_coeffs, s):
        """Solve the linear system to find the coefficients p_hat."""
        A = np.zeros((s, s), dtype=complex)
        b = -fourier_coeffs[s:2*s]

        for j in range(s):
            A[j] = fourier_coeffs[s+j-1::-1][:s]

        p_hat = np.linalg.solve(A, b)
        p_hat = np.concatenate(([1], p_hat))  # Include p̂(0) = 1
        return p_hat

    def identify_support(self, p_hat, N):
        """Identify the support by finding the roots of the polynomial."""
        roots = np.roots(np.flip(p_hat))
        support = np.round(np.angle(roots) * N / (2 * np.pi)).astype(int) % N
        return np.sort(support)

    def reconstruct_values_at_support(self, fourier_coeffs, support, N):
        M = len(fourier_coeffs)
        s = len(support)
        
        A = np.zeros((M, s), dtype=complex)
        for i in range(M):
            for j in range(s):
                A[i, j] = np.exp(-2j * np.pi * i * support[j] / N)  # Notice the negative exponent
        
        # Solve the overdetermined system A * x_support = fourier_coeffs
        x_support = np.linalg.lstsq(A, fourier_coeffs, rcond=None)[0]
        
        # Construct the full sparse vector
        x_reconstructed = np.zeros(N, dtype=complex)
        # x_reconstructed = np.zeros(N, dtype=int)

        x_reconstructed[support] = x_support

        x_reconstructed_round = np.round(x_reconstructed.real).astype(int)
        return x_reconstructed_round