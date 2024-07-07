% This is for debugging
% Define parameters (example values, replace with actual data)
n = 1; % number of countries i
m = 4; % number of countries j
p = 4; % number of sectors s

sigma; %= 3.57; % sigma_s
alpha; %7914.8 / 159327.1; % alpha_ijs
tau; %= 1e-10 + 1; % tau_ijs
X_hat; %= 36260000; % X_j
L = 161083000; % L_j
t = 1; % t_ijs
T = 192566; % T_ijs
delta = 1; % delta_is
gamma = 1; % gamma_ijs (compute as needed)

% Initial guesses for unknowns
w_hat = 19711; % initial guess for w_hat
X_hat_new = rand(m, 1); % initial guess for X_hat_new
P_hat_new = (6122 - 5972) / 5972 * 100; % initial guess for P_hat_new
pi_hat = rand(n, p); % initial guess for pi_hat

tol = 1e-6; % convergence tolerance
max_iter = 10; % maximum number of iterations

for iter = 1:max_iter
    % Equation (11): Update w_hat
    w_hat_new = zeros(n, 1);
    for i = 1:n
        sum_term = 0;
        for s = 1:p
            sum_term = sum_term + delta(i, s) * pi_hat(i, s);
        end
        w_hat_new(i) = sum_term;
    end

    % Equation (12): Update P_hat_new
    P_hat_new = zeros(m, p);
    for j = 1:m
        for s = 1:p
            sum_term = 0;
            for i = 1:n
                sum_term = sum_term + gamma(i, j, s) * (w_hat_new(i) * tau)^(1 - sigma);
            end
            P_hat_new(j, s) = sum_term^(1 / (1 - sigma));
        end
    end

    % Equation (13): Update X_hat_new
    X_hat_new = zeros(m, 1);
    for j = 1:m
        sum_term1 = (w_hat_new(j) * L / X_hat) * w_hat_new(j);
        sum_term2 = 0;
        for i = 1:n
            for s = 1:p
                sum_term2 = sum_term2 + (t * T / X_hat) * t * (w_hat_new(i))^(1 - sigma) * (P_hat_new(j, s))^(sigma - 1) * (tau)^(-sigma) * X_hat;
            end
        end
        sum_term3 = 0;
        for s = 1:p
            sum_term3 = sum_term3 + (pi_hat(j, s) / X_hat) * pi_hat(j, s);
        end
        X_hat_new(j) = sum_term1 + sum_term2 + sum_term3;
    end

    % Equation (10): Update pi_hat
    pi_hat_new = zeros(n, p);
    for i = 1:n
        for s = 1:p
            sum_term = 0;
            for j = 1:m
                sum_term = sum_term + alpha * (tau)^(-sigma) * (w_hat_new(i))^(1 - sigma) * (P_hat_new(j, s))^(sigma - 1) * X_hat_new(j);
            end
            pi_hat_new(i, s) = sum_term;
        end
    end

    % Debugging: Track values to identify where NaNs appear
    fprintf('Iteration %d\n', iter);
    disp('w_hat_new:');
    disp(w_hat_new);
    disp('P_hat_new:');
    disp(P_hat_new);
    disp('X_hat_new:');
    disp(X_hat_new);
    disp('pi_hat_new:');
    disp(pi_hat_new);

    % Check for convergence
    if norm(w_hat - w_hat_new) < tol && norm(X_hat_new - X_hat) < tol && norm(P_hat_new - P_hat_new) < tol && norm(pi_hat - pi_hat_new) < tol
        break;
    end

    % Update variables for next iteration
    w_hat = w_hat_new;
    X_hat = X_hat_new;
    P_hat_new = P_hat_new;
    pi_hat = pi_hat_new;
end

% Display the solutions
disp('w_hat solution:');
disp(w_hat);
disp('X_hat_new solution:');
disp(X_hat_new);
disp('P_hat_new solution:');
disp(P_hat_new);
disp('pi_hat solution:');
disp(pi_hat);

