module NoDisturbance
using LazySets
using Convex
using SCS
using Plots;
solver = SCSSolver(verbose=0)

function normc_hypersphere(x)
    return sign.(x) .* sqrt.(x.^2 ./ sum(x.^2))
end

using LinearAlgebra
n = 4
timestep = 0.2
A = [0 -1; 1 0]
A_d = exp(A*timestep)
upperBound = [1; 1]
lowerBound = [-1; -1]
genUser = [1 0; 0 1];
genRand = normc_hypersphere(randn(2,n-size(genUser,2)));
#genRand = genUser * [1/sqrt(2) -1/sqrt(2);1/sqrt(2) 1/sqrt(2)] # 4 generators
generators = hcat(genUser, genRand);
Gammy = Variable(n,1)
Center = Variable(2,1)
problem = maximize(dot(ones(n,1), Gammy) )
problem.constraints += Gammy >= 0
for t = 0:2*pi/timestep
            problem.constraints += A_d^t * Center - abs.( A_d^t * generators ) * Gammy >= lowerBound;
            problem.constraints += A_d^t * Center + abs.( A_d^t * generators ) * Gammy <= upperBound;
end
solve!(problem,solver)
problem.status
problem.optval
print(Gammy.value)
print(Center.value)

scaledRotatedGenerators = ( A_d^0 * generators ) .* ( ones(2,1) * Gammy.value' );
rotatedCenter = A_d^0 * Center.value;
l = [-1, -1];
h = [1, 1];
H = Hyperrectangle(low=l, high=h)
plot(H,fillalpha=0, line=0.5, linecolor = :blue)

for t = 0:2*pi/timestep
    scaledRotatedGenerators = ( A_d^t * generators ) .* ( ones(2,1) * Gammy.value' );
    rotatedCenter = A_d^t * Center.value;
    Z = Zonotope(vec(rotatedCenter), scaledRotatedGenerators)
    plot!(Z, fillalpha=0.1, line=0.5)
end
Z = Zonotope(vec(rotatedCenter),scaledRotatedGenerators);
p = plot!(Z, fillalpha=0, line=5, linecolor = :blue)

display(p)
end  # moduleNoDisturbance
