module WithDisturbance3D
using LazySets
using Convex
using SCS
using Plots;
solver = SCSSolver(verbose=0)

function normc_hypersphere(x)
    return sign.(x) .* sqrt.(2* x.^2 ./ sum(x.^2))
end

using LinearAlgebra
function main(n,timestep)
    A = [0 -1 0; 1 0 0; 0 0 1]
    A_d = exp(A*timestep)
    upperBound = [1; 1; 1]
    lowerBound = [-1; -1 ; -1]
    genUser = [1 0 0; 0 1 0; 0 0 1];
    genRand = normc_hypersphere(randn(3,n-size(genUser,2)));
    generators = normc_hypersphere(rand(3,n))
    disturbanceGenerators = [0.05 0 0; 0 0.05 0; 0 0 0];
    disturbanceCenter = [0; 0; 0];
    cenDist = [0 0 0 ; 0 0 0 ; 0 0 0];
    genDist = [0 0 0 ; 0 0 0 ; 0 0 0];
    Gammy = Variable(n,1)
    Center = Variable(3,1)
    problem = maximize(dot(ones(n,1), Gammy) )
    problem.constraints += Gammy >= 0
    for t = 0:2*pi/timestep
                problem.constraints += A_d^t * Center + cenDist * ones( size( cenDist, 2 ), 1 ) - abs.( A_d^t * generators ) * Gammy - abs.( genDist ) * ones( size( genDist, 2 ), 1 ) >= lowerBound;
                problem.constraints += A_d^t * Center + cenDist * ones( size( cenDist, 2 ), 1 ) + abs.( A_d^t * generators ) * Gammy + abs.( genDist ) * ones( size( genDist, 2 ), 1 ) <= upperBound;
                cenDist = hcat( A_d * cenDist, disturbanceCenter );
                genDist = hcat( A_d * genDist, disturbanceGenerators );
    end
    solve!(problem,solver)
    print(Gammy.value)
    print(Center.value)
    problem.status
    problem.optval

    scaledRotatedGenerators = ( A_d^0 * generators ) .* ( ones(3,1) * Gammy.value' );
    rotatedCenter = A_d^0 * Center.value;
    l = [-1, -1];
    h = [1, 1];
    H = Hyperrectangle(low=l, high=h)
    p = plot(H,fillalpha=0, line=0.5, linecolor = :blue)


    cenDist = [0 0 0 ; 0 0 0 ; 0 0 0];
    genDist = [0 0 0 ; 0 0 0 ; 0 0 0];
    for t = 0:2*pi/timestep
        scaledRotatedGenerators = ( A_d^t * generators ) .* ( ones(3,1) * Gammy.value' );
        rotatedCenter = A_d^t * Center.value;
        cenDist = hcat( A_d * cenDist, disturbanceCenter );
        genDist = hcat( A_d * genDist, disturbanceGenerators );
        c = vec(sum(hcat(rotatedCenter, cenDist), dims =2));
        g = hcat(scaledRotatedGenerators, genDist);
        G= g[:, vec(mapslices(col -> any(col.> 0.001), g, dims = 1))]; #reduce null columns
        projc = c[1:2]
        projG = G * [ 1 0 0; 0 1 0; 0 0 0]
        Z = Zonotope(projc,projG);
        #projZ = Approximations.project(Z,[1; 2], HPolygon, 3)
        plot!(Z, fillalpha=0.01, line=0.5)
    end

    scaledRotatedGenerators = ( A_d^0 * generators ) .* ( ones(3,1) * Gammy.value' );
    rotatedCenter = A_d^0 * Center.value;
    Srg = scaledRotatedGenerators[:, vec(mapslices(col -> any(col.> 0.001), scaledRotatedGenerators, dims = 1))]
    Proj = Srg * projG
    projc = vec(rotatedCenter)[1:2]
    Z = Zonotope(projc,Proj);
    #projZ = Approximations.project(Z,[1; 2], HPolygon, 3) #write a projection routine for zonotope.
    plot!(Z, fillalpha=0.1, line=5)

    display(p)
end
main(10, 20)
end  # moduleNoDisturbance
