module ZonoVolume
# There are two fuctions in this module
#volume(Z) calculates the volume of the zonotope Z by its fomula
#approximate(Z,n) approximates the volume of the zonotope Z by sampling n points
using LazySets
using LinearAlgebra
using Combinatorics
export zonoVolume
# input matrix A of size n*k, return the list of ordered submatrix of size n*r, where r is the rank
function subMatrix(A)
    k = size(A)[2]
    n = size(A)[1]
    r = rank(A)
    result = []
    index = collect(combinations(1:k,r))
    for i = 1: size(index)[1]
        push!(result, A[:,index[i]])
    end
    return result
end
#imput a generating matrix A of a zonotope Z, returns the volumn
function zonoVolume(Z)
   A = Z.generators
   sum = 0
       for i = 1:size(subMatrix(A))[1]
           S = subMatrix(A)[i]
           sum = sum + sqrt(det(S'*S))
       end
   return sum*2^(size(A)[1])
end

volume(Z) = print(zonoVolume(Z))

using Distributions
function randomVertices(topRight, s) #generate s random vertices within the bounded box with top right corner
    dim = size(topRight)[1]
    result = []
    for j = 1:s
        temp = Array{Float64}(undef, 1, dim)
        for i = 1:dim
            temp[i] = rand(Uniform(-topRight[i],topRight[i]),1)[1]
        end
        push!(result, temp)
    end
    return result
end

function zonoVolumeApprox(Z,sampleSize) #approximate the volume by taking certain size of sample
    A = Z.generators
    k = size(A)[2]
    n = size(A)[1]
    r = rank(A)
    topRight = sum(broadcast(abs, A), dims = 2)
    boxVolumn = prod(topRight) * 2^n
    sample = randomVertices(topRight, sampleSize)
    percentage = sampleInZono(sample, Z);
    return percentage * boxVolumn
end

#calculate the percentage of sample vertices that are in the zonotope
function sampleInZono(sample, Z)
    count = 0;
    for i = 1:size(sample)[1]
        if ∈(vec(sample[i]), Z)
            count = count + 1
        end
    end
    return count/size(sample)[1]
end
approximate(Z,n) = print(zonoVolumeApprox(Z, n))

using Plots; gr()
xs = 1 : 1 : 20
vol = []
approx = []
for i = 1:20
    Z = rand(Zonotope; dim = 6)
    push!(vol, zonoVolume(Z))
    push!(approx, zonoVolumeApprox(Z,10000))
end
data = [vol, approx]
labels = ["Zonotope volume" "Approiximation"]

p = plot(xs, data, label = labels, linewidth=2, markersize = 10, title="dim = 6 Zonotope volume and approximation")
display(p)
end # module
