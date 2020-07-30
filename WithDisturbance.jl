module WithDisturbance
using LazySets
using Convex
using SCS
using Plots;
using LinearAlgebra

solver = SCSSolver(verbose=0)

function normc_hypersphere(x)
    return  sign.(x) .* sqrt.(x.^2 ./ sum(x.^2, dims = 1))
end


function example1()
    n = 20
    timestep= 0.5
    genUser = [1 0; 0 1];
    genRand = normc_hypersphere(randn(2,n-size(genUser,2)));
    generators = hcat(genUser, genRand)
    disturbanceGenerators = [0.05 0; 0 0.05];
    disturbanceCenter = [0; 0];
    main(n,timestep, generators, disturbanceGenerators, disturbanceCenter)
end


function main(n,timestep, generators, disturbanceGenerators, disturbanceCenter)
    A = [0 -1; 1 0]
    A_d = exp(A*timestep)
    upperBound = [1; 1]
    lowerBound = [-1; -1]
    cenDist = [0 0; 0 0];
    genDist = [0 0; 0 0];
    Gammy = Variable(n,1)
    Center = Variable(2,1)
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

    l = [-1, -1];
    h = [1, 1];
    H = Hyperrectangle(low=l, high=h)
    p = plot(H,fillalpha=0, line=0.5, linecolor = :blue)

    cenDist = [0 0; 0 0];
    genDist = [0 0; 0 0];
    for t = 0:2*pi/timestep
        scaledRotatedGenerators = ( A_d^t * generators ) .* ( ones(2,1) * Gammy.value' );
        rotatedCenter = A_d^t * Center.value;
        cenDist = hcat( A_d * cenDist, disturbanceCenter );
        genDist = hcat( A_d * genDist, disturbanceGenerators );
        c = vec(sum(hcat(rotatedCenter, cenDist), dims =2));
        g = hcat(scaledRotatedGenerators, genDist);
        G= g[:, vec(mapslices(col -> any(col.> 0.001), g, dims = 1))]; #reduce null columns
        Z = Zonotope(c,g);
        plot!(Z, fillalpha=0.1, line=0.5)
    end

    scaledRotatedGenerators = ( A_d^0 * generators ) .* ( ones(2,1) * Gammy.value' );
    rotatedCenter = A_d^0 * Center.value;
    Srg = scaledRotatedGenerators[:, vec(mapslices(col -> any(col.> 0.001), scaledRotatedGenerators, dims = 1))]
    Z = Zonotope(vec(rotatedCenter),Srg);
    plot!(Z, fillalpha=0.1, line=5)

    display(p)
end

end  # moduleNoDisturbance
