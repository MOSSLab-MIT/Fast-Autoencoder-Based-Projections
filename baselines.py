import torch
import torch.optim as optim
import numpy as np
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

import data_generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Customize these based on your own needs
VALID_METHODS = [
    'projected_gradient',
    'penalty_method',
    'augmented_lagrangian',
    'interior_point'
]
VALID_CONSTRAINTS = [] # ensure the return format follows that of the data_generation.py file
VALID_OBJECTIVES  = ['qp', 'lp', 'distance']

# Data Generation for Different Objective Types
def generate_qp_problem(n_vars=2, batch_size=1):
    """Generate random QP problem: min 0.5 x^T Q x + p^T x"""
    A = torch.randn(batch_size, n_vars, n_vars, device=device)
    Q = torch.bmm(A.transpose(1, 2), A) + 0.01 * torch.eye(n_vars, device=device).unsqueeze(0)
    p = torch.randn(batch_size, n_vars, device=device)
    return Q, p

def generate_lp_problem(n_vars=2, batch_size=1):
    """Generate random LP problem: min c^T x"""
    c = torch.randn(batch_size, n_vars, device=device)
    return c

def generate_distance_problem(n_vars=2, batch_size=1):
    """Generate random distance minimization problem: min ||x - target||^2"""
    target = torch.randn(batch_size, n_vars, device=device) * 3.0  # Random targets
    return target

# Baselines
def compute_violation_score(x, shape_name, sigma=0.05, n_samples=32):
    batch_size, dim = x.shape
    eps = torch.randn(n_samples, batch_size, dim, device=x.device)
    noise = eps * sigma
    
    with torch.no_grad():
        x_samples = x.unsqueeze(0) + noise
        x_samples_flat = x_samples.reshape(-1, dim).cpu().numpy()
        feasible_flat = data_generation.check_feasibility(x_samples_flat, shape_name).reshape(-1)
        
        violation_bool = torch.tensor(~feasible_flat, dtype=torch.float32, device=x.device)
        violation_matrix = violation_bool.view(n_samples, batch_size) 
        
        violation_score = violation_matrix.mean(dim=0) 
        violation_expanded = violation_matrix.unsqueeze(-1)         
        es_grad = (violation_expanded * eps / sigma).mean(dim=0)    
        
    differentiable_score = violation_score + (x * es_grad).sum(dim=-1) - (x.detach() * es_grad).sum(dim=-1)
    return differentiable_score

# Projected Gradient
def solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = 0.5 * (x * torch.matmul(Q, x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p * x).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

def solve_lp_with_projection(c, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = (c * x).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

def solve_distance_with_projection(target, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = ((x - target) ** 2).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

def penalty_method(objective_fn, x_init, shape_name, penalty_coeff=10.0, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    optimizer = optim.Adam([x], lr=0.05)
    for iter_num in range(max_iter):
        optimizer.zero_grad()
        obj = objective_fn(x).mean()
        score = compute_violation_score(x, shape_name)
        loss = obj + penalty_coeff * score.mean()
        loss.backward()
        optimizer.step()
        if iter_num > 0 and iter_num % 20 == 0:
            penalty_coeff *= 2.0
    return x.detach()

def augmented_lagrangian(objective_fn, x_init, shape_name, outer_iter=10, inner_iter=10):
    x = x_init.clone().requires_grad_(True)
    lambda_dual = torch.zeros(x.shape[0], device=x.device)
    rho = 10.0
    optimizer = optim.Adam([x], lr=0.05)
    for _ in range(outer_iter):
        for _ in range(inner_iter):
            optimizer.zero_grad()
            obj = objective_fn(x)
            violation = compute_violation_score(x, shape_name)
            loss = obj.mean() + (lambda_dual * violation).mean() + 0.5 * rho * (violation ** 2).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_violation = compute_violation_score(x, shape_name)
            lambda_dual = lambda_dual + rho * final_violation
    return x.detach()

def interior_point_method(objective_fn, x_init, shape_name, max_iter=100, tau=0.1):
    x = x_init.clone().requires_grad_(True)
    optimizer = optim.Adam([x], lr=0.01)
    barrier_coeff = 0.5
    for iter_num in range(max_iter):
        optimizer.zero_grad()
        obj = objective_fn(x)
        violation_prob = compute_violation_score(x, shape_name, sigma=0.1)
        slack = tau - violation_prob
        barrier = -barrier_coeff * torch.log(torch.clamp(slack, min=1e-6))
        loss = obj.mean() + barrier.mean()
        loss.backward()
        optimizer.step()
        if iter_num > 0 and iter_num % 20 == 0:
            barrier_coeff *= 0.5
    return x.detach()

# Main Testing Function
def run_tests():
    num_seeds = 5
    num_problems_per_seed = 300

    print("Starting Baseline Methods Testing")
    for shape_name in VALID_CONSTRAINTS:
        for obj_type in VALID_OBJECTIVES:
            print(f"\nTesting Shape: {shape_name} | Objective: {obj_type}")
            print("-" * 60)
            
            method_results = {m: {'objectives': [], 'violations': [], 'times': [], 'optimality_gaps': []} for m in VALID_METHODS}

            for seed in range(num_seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                problems = []
                ground_truth_objectives = []

                # Generate problems
                for _ in range(num_problems_per_seed):
                    x_init = torch.randn(1, 2, device=device)
                    if obj_type == 'qp':
                        Q, p = generate_qp_problem(n_vars=2, batch_size=1)
                        problems.append({'Q': Q, 'p': p})
                        x_true = solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=100)
                        gt_obj = 0.5 * (x_true * torch.matmul(Q.to(x_true.dtype), x_true.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p.to(x_true.dtype) * x_true).sum(dim=1)
                    elif obj_type == 'lp':
                        c = generate_lp_problem(n_vars=2, batch_size=1)
                        problems.append({'c': c})
                        x_true = solve_lp_with_projection(c, x_init, shape_name, max_iter=100)
                        gt_obj = (c.to(x_true.dtype) * x_true).sum(dim=1)
                    else:
                        target = generate_distance_problem(n_vars=2, batch_size=1)
                        problems.append({'target': target})
                        x_true = solve_distance_with_projection(target, x_init, shape_name, max_iter=100)
                        gt_obj = ((x_true - target.to(x_true.dtype)) ** 2).sum(dim=1)
                    
                    ground_truth_objectives.append(gt_obj.item())

                # Test methods
                for method in VALID_METHODS:
                    for i, problem in enumerate(tqdm(problems, desc=method, leave=False)):
                        start_time = time.time()
                        x_init = torch.randn(1, 2, device=device)

                        if obj_type == 'qp':
                            Q, p = problem['Q'], problem['p']
                            objective_fn = lambda x: 0.5 * (x * torch.matmul(Q.to(x.dtype), x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p.to(x.dtype) * x).sum(dim=1)
                        elif obj_type == 'lp':
                            c = problem['c']
                            objective_fn = lambda x: (c.to(x.dtype) * x).sum(dim=1)
                        else:
                            target = problem['target']
                            objective_fn = lambda x: ((x - target.to(x.dtype)) ** 2).sum(dim=1)

                        if method == 'projected_gradient':
                            if obj_type == 'qp': x_sol = solve_qp_with_projection(Q, p, x_init, shape_name)
                            elif obj_type == 'lp': x_sol = solve_lp_with_projection(c, x_init, shape_name)
                            else: x_sol = solve_distance_with_projection(target, x_init, shape_name)
                        elif method == 'penalty_method': 
                            x_sol = penalty_method(objective_fn, x_init, shape_name)
                        elif method == 'augmented_lagrangian': 
                            x_sol = augmented_lagrangian(objective_fn, x_init, shape_name)
                        elif method == 'interior_point': 
                            x_sol = interior_point_method(objective_fn, x_init, shape_name)

                        end_time = time.time()
                        obj_value = objective_fn(x_sol).item()
                        gt_obj_value = ground_truth_objectives[i]
                        optimality_gap = abs(obj_value - gt_obj_value)
                        
                        x_np = x_sol.detach().cpu().numpy()
                        is_feasible = data_generation.check_feasibility(x_np, shape_name)[0]

                        method_results[method]['objectives'].append(obj_value)
                        method_results[method]['optimality_gaps'].append(optimality_gap)
                        method_results[method]['violations'].append(0 if is_feasible else 1)
                        method_results[method]['times'].append(end_time - start_time)
            # Print Metrics Summary
            print(f"{'Method':<25} | {'Feasibility':>12} | {'Time (ms)':>10} | {'Opt. Gap':>12}")
            for method in VALID_METHODS:
                feas_rate = 1.0 - np.mean(method_results[method]['violations'])
                avg_time = np.mean(method_results[method]['times']) * 1000
                avg_gap = np.mean(method_results[method]['optimality_gaps'])
                print(f"{method:<25} | {feas_rate:>12.1%} | {avg_time:>10.2f} | {avg_gap:>12.4f}")

if __name__ == "__main__":
    run_tests()