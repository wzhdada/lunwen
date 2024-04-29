# Figure 5.3a in "Infinite GMRES for parameterized linear systems"

using SparseArrays
using LinearAlgebra
using PyPlot
using MAT
pygui(true)
include("iar.jl");
include("infgmres.jl");

# Problem represented with explicitly available derivatives
struct GeneralProblem
    sigma
    Aders
    Afun
    b
end

function compute_Mlincomb(prob::GeneralProblem,sigma,Y)
    if (prob.sigma != sigma)
        @show sigma
        @show prob.sigma
        error("Wrong sigma");
    end
    z=zeros(eltype(Y),size(Y,1));
    for i=1:size(Y,2)
        z += prob.Aders[i]*Y[:,i];
    end
    return z;
end
function compute_Mder(prob::GeneralProblem,mu)
    return prob.Afun(mu);
end


## Define problem
data=matread("delay_problem_matrices.mat")
A0=data["A0"];
A1=data["A1"];
b=vec(data["b"]);
n=length(b);

# Identity matrix
II=Matrix{Float64}(I,n,n);

Afun= s-> -s*II+A0+A1*exp(-s);
sigma=0;
N=100;
ders=Vector{Matrix{Float64}}(undef,N);

for i=1:N
    ders[i]=((-1)^(i+1))*A1;
end
ders[1] += A0;
ders[2] += -II;

myproblem=GeneralProblem(sigma,ders,Afun,b);



## Run Infinite GMRES
(f,hist1)=infgmres(myproblem,30,method=:iar);
mu_list = range(1.e-5,stop=1.e-1,length=20);
mu_list = mu_list[2:end];
how_many = zeros(length(mu_list),1);
tol = 1.e-12;

for i=1:length(mu_list)
    global mu
    mu = mu_list[i];
    A_of_mu = myproblem.Afun(mu);
    res = zeros(100,1);
    for m=1:200
        global f
        (xx)=f(mu,m)
        res[m] = norm(A_of_mu*xx - myproblem.b)/norm(myproblem.b);
        if res[m] < tol
            how_many[i] = m;
            break;
        end
    end
end

figure(1)
ax = PyPlot.gca()
xlabel("mu")
ylabel("its")
plot(mu_list,how_many,"ro")
