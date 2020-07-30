module volumeConvex
## using convex.jl to run the optimization (not successful)
using LazySets
using LinearAlgebra
using Combinatorics
using LinearAlgebra
using Convex
using GLPKMathProgInterface
#using SCS
#solver = SCSSolver(verbose=0)
# input matrix A of size n*k, return the list of ordered submatrix of size n*r, where r is the rank
function subMatrix(A)
    k = size(A)[2]
    n = size(A)[1]
    r = n#rank(A)
    result = []
    index = collect(combinations(1:k,r))
    for i = 1: size(index)[1]
        Ai = A[:,index[i]]
        push!(result, Ai)
    end
    return result
end
#imput a generating matrix A of a zonotope Z, returns the volumn
function zonoVolume(A)
   result = 0
       for i = 1:size(subMatrix(A))[1]
           S = subMatrix(A)[i]
           result = result + 1/2 * logdet(S'*S)
       end
   return result *2^(size(A)[1])
end

function normc_hypersphere(x)
    return sign.(x) .* sqrt.(x.^2 ./ sum(x.^2))
end


function main()
    n = 2
    timestep = 0.2
    A = [0 -1; 1 0]
    A_d = exp(A*timestep)
    upperBound = [1; 1]
    lowerBound = [-1; -1]
    genUser = [1 0; 0 1];
    genRand = normc_hypersphere(randn(2,n-size(genUser,2)));
    #generators = hcat(genUser, genRand);
    generators = genUser
    disturbanceGenerators = [0.01 0; 0 0.01];
    disturbanceCenter = [0 ; 0];
    cenDist = [0 0; 0 0];
    genDist = [0 0; 0 0];
    Gammy = Variable(n,1)
    Center = Variable(2,1)
    #Z = Zonotope(vec(rotatedCenter),scaledRotatedGenerators);
    scaledRotatedGenerators = generators .* ( ones(2,1) * Gammy' )
    obj = zonoVolume(scaledRotatedGenerators)
    problem = maximize(obj)
    problem.constraints += Gammy >= 0
    for t = 0:2*pi/timestep
        problem.constraints += A_d^t * Center + cenDist * ones( size( cenDist, 2 ), 1 ) - abs.( A_d^t * generators ) * Gammy - abs.( genDist ) * ones( size( genDist, 2 ), 1 ) >= lowerBound;
        problem.constraints += A_d^t * Center + cenDist * ones( size( cenDist, 2 ), 1 ) + abs.( A_d^t * generators ) * Gammy + abs.( genDist ) * ones( size( genDist, 2 ), 1 ) <= upperBound;
        cenDist = hcat( A_d * cenDist, disturbanceCenter );
        genDist = hcat( A_d * genDist, disturbanceGenerators );
    end
    solve!(problem, GLPKSolverMIP())
    print(Gammy.value)
    print(Center.value)
    problem.status
    problem.optval
end
main()
end  # modue volumeOptimization
