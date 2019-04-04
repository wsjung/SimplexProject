%% test case 1 (in-class example)
%c = [2;4;0;0;0];
%A = [4,6,1,0,0; 2,6,0,1,0; 0,1,0,0,1];
%b = [120;72;10];
%% test case 2 (ex 8.1 from textbook)
% c = [13;5;0;0;0];
% A = [4,1,1,0,0;1,3,0,1,0;3,2,0,0,1];
% b = [24;24;23];
% Simplex(c,A,b)
%%

%
% simplex
% ---------
% c = (n x 1 column vector)
% A = (m x n matrix)
% b = (m x 1 column vector)
%
function sol = Simplex(c,A,b)
%SIMPLEX Uses the simplex algorithm to solve an input linear program. User
%must input a vector c, matrix A, and vector b. If the linear program
%inputs are not valid an error message will be given.
%
% sol = simplex(c,A,b) returns the optimal solution (as a vector) to the given linear
%
%A vector c that represents the vector of coefficients of the objective function of
%the linear program
%
%A matrix A that represents the coefficient matrix of the linear program
%
%A vector b that represents a vector of scalars that represents the right hand side
%to the coefficient matrix of the linear program
%
%   See also solve
%
dim_c = size(c); % expected: [n,1]
dim_A = size(A); % expected: [m,n]
dim_b = size(b); % expected: [m,1]

% check for dimension compatibiility
if isCompat(dim_c,dim_A,dim_b)
    % extract m and n 
    m = dim_A(1);
    n = dim_A(2);
    
    sol = solve(c,A,b,n,m);
else
    % throws informative error if the dimensions are not compatible
    error_msg = 'Error: Incompatible dimensions.\nMake sure the following are satisfied:\n\tsimplex(c,A,b), where\n\t\tc = (n x 1) column vector\n\t\tA = (m x n) matrix\n\t\tb = (m x 1) column vector';
    error(sprintf(error_msg)) % had to nest bc error() wouldn't parse newlines
end
end

function bool = isCompat(dim_c,dim_A,dim_b)
%ISCOMPAT checks dimension compatibility of input matrix & vectors
%
% bool = isCompact(dim_c, dim_A, dim_b) determines the dimensions compatibility
%and returns whether it is true or false if the dimensions of
% the input vectors c, and b, and the matrix A are compatible. 
%
%expected dimensions of c are: [n,1]
%expected dimensions of A are: [m,n]
%expected dimensions of b are: [m,1]

bool = (dim_c(2)==1 && dim_b(2)==1 && dim_c(1)==dim_A(2) && dim_b(1)==dim_A(1));
end


function origin = calcOrigin(b,n,m)
%CALCORIGIN  computes the origin of the linear program
%
%origin = calcOrigin(b,n,m) looks at the vector b and determines/returns the origin of the linear program
%
origin = zeros(n,1); % first n-m are zero
for i = (n-m+1):n
    %i
    %i-(n-m)
    origin(i,1) = b(i-(n-m),1);
end
end


function [B,N] = calcBases(A,n,m,basics,nonbasics)
%CALCBASES calculates basis B and matrix N given input arguments
%
% [B, N] = calcBases(A,n,m, basics, nonbasics) determines which 
% coefficients from the coefficient matrix A corresponds to the basic
% variables puts them in a matrix B  and determines which b coefficients from 
%the coefficient matrix A correspond to the nonbasic variables and puts them
%in a matrix N. Returns B and N.
%
B = []; % basis
for i=1:m
    B = [B,A(:,basics(i))];
end
%B
%nonbasis
N = [];
for i=1:n-m
    N = [N,A(:,nonbasics(i))];
end
%N
end


function [c_B,c_N] = calcObj(c,n,m,basics,nonbasics)
%CALCOBJ calculates objective function with respect to basic (c_B) / nonbasic
% variables (c_N)
%
%[c_B,c_N] = calcObj(c,n,m,basics,nonbasics) looks at the vector c and takes the
%coefficients (from the objective function) corresponding to the basic variables and puts them 
%in a vector c_b and then takes the coefficients 
%(from the objective function) corresponding to the
%nonbasic variables and puts them in a vector c_N. Returns the two vectors c_B and c_N.
%
c_B = [];
c_N = [];
for i=1:m
    c_B(i,1) = c(basics(i),1);
end
for i=1:n-m
    c_N(i,1) = c(nonbasics(i),1);
end
%c_B
%c_N
end


function final_sol = solve(c,A,b,n,m)
%SOLVE This is the main simplex algorithm
%
%final_sol = solve(c,A,b,n,m) This function follows and actually solbes the input program
%using the simplex algorithm
%and returns the final optimal solution

%   See also calcOrigin, calcBases, calcObj
%% Step 0: Initialization
% initial solution is the origin
sol = calcOrigin(b,n,m);

basics = [n-m+1:n]; %indices of m basic vars
nonbasics = [1:n-m]; %indices of n-m nonbasic vars

% basis B and nullspace N
[B,N] = calcBases(A,n,m,basics,nonbasics);

% objective function coefficients associated with basic/nonbasic vars
[c_B,c_N] = calcObj(c,n,m,basics,nonbasics);

while true
    %% Step 1: Compute Simplex Multipliers and Vector of Reduced Costs
    % simplex multipliers y
    y = transpose(transpose(c_B)/B);
    
    % calculate reduced costs
    red_cost = transpose(transpose(c_N) - transpose(y)*N);
    
    simp_dir = [];
    entering = -1; % initializations %
    leaving = -1;
    
    %% Step 2: Optimality check
    if red_cost <= 0
        % current solution is optimal, then it is the final solution
        final_sol = sol;
        return;
    else
        %% Step 3: Compute simplex direction
        % find first nonbasic variable with reduced cost > 0 (bland's rule)
        for i=1:n-m
            if red_cost(i) > 0
                entering = i;
                % calculate simplex direction
                simp_dir = B\(-A(:,nonbasics(entering)));
                break;
            end
        end
        %% Step 4: Compute maximum step size
        % check for unboundedness
        if simp_dir >= 0
            %UNBOUNDED, return an error message because linear program has
            %an infeasible solution
            error_msg = "unbounded";
            error(sprintf(error_msg));
            return;
        else
            % calculate max step size using Ratio Test
            step_sizes = sol(basics(1:m))./-simp_dir;
            % max step size and index of leaving variable associated with it
            delta_max = min(step_sizes(step_sizes > 0)); % only for step_size > 0
            
            for i=1:m
                if step_sizes(i) == delta_max;
                   leaving = i;
                   break;
                end
            end
            
            entering_var = nonbasics(entering);
            leaving_var = basics(leaving);
            
            temp = zeros(n,1);
            temp(entering_var,1) = delta_max; % entering variable
            
            % full simplex direction
            for i=1:m
                temp(basics(i),1) = delta_max*simp_dir(i);
            end
            
            %% Step 5: Update Solution and Basis
            
            % update solution
            sol = sol + temp;
            
            % update basic and nonbasic variables
            temp = basics(leaving);
            basics(leaving) = nonbasics(entering);
            nonbasics(entering) = temp;
            
            % update B and N
            [B,N] = calcBases(A,n,m,basics,nonbasics);
            
            % update c_B & c_N
            [c_B,c_N] = calcObj(c,n,m,basics,nonbasics);
            
        end
    end
end % end while %
end % end function %
