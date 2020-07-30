##Using JuMP to run optimization.
using JuMP, Ipopt, Test
using LazySets
using LinearAlgebra
using Combinatorics
using Plots
function zonoVolume(A, Gamma::Array{VariableRef,1})
   result = 0
   k = size(A)[2]
   n = size(A)[1]
   r = n #r is the rank, here we assume generators matrix is full rank
   index = collect(combinations(1:k,r))
       for i = 1: size(index)[1]
           S = A[:,index[i]]
           result = result + sqrt(det(S'*S)) * prod(Gamma[index[i]]) ## Is this equvialent to the true volume?
       end
   return result*2^(size(A)[1])
end

function normc_hypersphere(x)
    return sign.(x) .* sqrt.(x.^2 ./ sum(x.^2))
end

function main(; verbose = true)
    n = 10
    timestep = 0.2
    A = [0 -1; 1 0]
    A_d = exp(A*timestep)
    upperBound = [1; 1]
    lowerBound = [-1; -1]
    genUser = [1 0; 0 1];
    genRand = randn(2,n-size(genUser,2));
    generators = normc_hypersphere(hcat(genUser, genRand));

    model = Model(with_optimizer(Ipopt.Optimizer))

    @variable(model, Gammy[1:n]>=0)
    @variable(model, Center[1:2])

    #@objective(model, Max, dot(ones(n,1), Gammy)) #this is the original herustic
    @objective(model, Max, zonoVolume(generators, Gammy))

    for t = 0:2*pi/timestep
        @constraint(model, A_d^t * Center - abs.( A_d^t * generators ) * Gammy .>= lowerBound);
        @constraint(model, A_d^t * Center + abs.( A_d^t * generators ) * Gammy .<= upperBound);
    end


    print(model)


    JuMP.optimize!(model)

    obj_value = JuMP.objective_value(model)
    Gammy_value = JuMP.value.(Gammy)
    Center_value = JuMP.value.(Center)

    println("Objective value: ", obj_value)
    println("Gammy = ", Gammy_value)
    println("Center = ", Center_value)


    l = [-1, -1];
    h = [1, 1];
    H = Hyperrectangle(low=l, high=h)
    p = plot(H,fillalpha=0, line=0.5, linecolor = :blue)

    for t = 0:2*pi/timestep
        scaledRotatedGenerators = ( A_d^t * generators )* Diagonal(Gammy_value)
        rotatedCenter = A_d^t * Center_value;
        Z = Zonotope(vec(rotatedCenter), scaledRotatedGenerators)
        plot!(Z, fillalpha=0.05, line=0.5)
    end

    scaledRotatedGenerators = generators * Diagonal(Gammy_value)
    rotatedCenter = Center_value;
    Z = Zonotope(vec(rotatedCenter),scaledRotatedGenerators);
    p = plot!(Z, fillalpha=0, line=5, linecolor = :red)

    display(p)
end
