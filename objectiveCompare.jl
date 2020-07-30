#the main() function implements check convexity between two scaling vectors using different objective functions
#the autoTest() implements automatic test routine to check convexity of a fucntion using finite difference
using LazySets
using LinearAlgebra
using Combinatorics
using Distributions
using Plots
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
#true zonotope volume
function zonoVolume(Z)
   A = Z.generators
   result = 0
       for i = 1:size(subMatrix(A))[1]
           S = subMatrix(A)[i]
           result = result + sqrt(det(S'*S))
       end
   return result*2^(rank(A))
end
#imput a generating matrix A of a zonotope Z, returns the volume herustic: sum of log
function zonoVolumeSumLog(Z)
   A = Z.generators
   sum = 0
       for i = 1:size(subMatrix(A))[1]
           S = subMatrix(A)[i]
           sum = sum + log(sqrt(det(S'*S)))
       end
   return sum #*2^(rank(A))
end
function newFun(A, Gamma)
   sum = 0
   k = size(A)[2]
   n = size(A)[1]
   r = n #rank(A)
   index = collect(combinations(1:k,r))
       for i = 1: size(index)[1]
           S = A[:,index[i]]
           sum = sum + det(S'*S) * prod(Gamma[index[i]])
       end
   return sum*2^(size(A)[1])
end
#check convexity between two scaling vectors using different objective functions
function main()

    dim = 4
    order = 8
    A = rand(dim,order)
    S1 = Diagonal(rand(Uniform(0,1),order,1)[:])
    S2 = Diagonal(rand(Uniform(0,1),order,1)[:])
    vol = []
    logSum = []
    sumLog = []
    sumGamma = []
    newFunc = []
    xs = 0 : 0.05 : 1
    for i in xs
        #G = i*A*S1 + (1-i) * A*S2
        S = i*S1 + (1-i) *S2
        G = A*S
        Z = Zonotope(zeros(dim), G)
        push!(sumLog, zonoVolumeSumLog(Z))
        push!(vol, zonoVolume(Z))
        push!(logSum, log(zonoVolume(Z)))
        push!(sumGamma,sum(diag(S)))
        push!(newFunc, newFun(A,diag(S)))
    end
    data = [vol]
    labels = ["Zonotope volume"]
    p1 = plot(xs, data, label = labels, linewidth=2, markersize = 10)
    display(p1)

    data1 = [logSum]
    labels1 = ["log volume"]
    p1 = plot(xs, data1, label = labels1, linewidth=2, markersize = 10)
    display(p1)
    data2 = [sumLog]
    labels2 = ["sum of Log"]
    p2 = plot(xs, data2, label = labels2, linewidth=2, markersize = 10)
    display(p2)
    data3 = [sumGamma]
    labels3 = ["sum of scale factors"]
    p3 = plot(xs, data3, label = labels3, linewidth=2, markersize = 10)
    display(p3)
    data4 = [newFunc]
    labels4 = ["new function"]
    p4 = plot(xs, data4, label = labels4, linewidth=2, markersize = 10)
    display(p4)
end
#second derivavtive test automate
function secondDiff(vol)
    result = []
    for i in 2:20
        push!(result,(vol[i + 1] - 2 * vol[i] + vol[i - 1]) / 4)
    end
    return result
end

function concav(diff)
    for i in diff
        if(i > 0.1)
            return false
        end
    end
    return true
end

function convex(diff)
    for i in diff
        if(i < 0)
            return false
        end
    end
    return true
end

function autoTest(dim, order, n)
    xs = 0 : 0.01 : 1
    for i in 1:n
        A = rand(dim,order)
        S1 = Diagonal(rand(Uniform(0,1),order,1)[:])
        S2 = Diagonal(rand(Uniform(0,1),order,1)[:])
        vol = []
        #logVol = []
        #conds = []
        for i in xs
            G = i*A*S1 + (1-i) * A*S2
            Z = Zonotope(zeros(dim), G)
            #push!(conds, log(cond(G)))
            push!(vol, zonoVolume(Z))
            #push!(logVol, log(zonoVolume(Z)))
        end
        diff = secondDiff(vol)
        if(!concav(diff))
                #&& (!convex(diff))
            data = [vol]
            labels = ["volumn"]
            ttl = "dim = " * string(dim) * ", order = " * string(order)
            p= plot(xs, data, label = labels, linewidth=2, markersize = 10, title = ttl)
            print(diff)
            print(labels)
            display(p)
        end
    end
end
#autoTest(6,8,50)

# for i in 3:6
#     for j in i:2*i
#        autoTest(i,j,20)
#        print(j)
#     end
#    print(i)
# end
