# Figure 5.2a "Infinite GMRES for parameterized linear systems"

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


## Check results
mu=0.01;
kk=12
errv=zeros(kk);
for k=1:kk
    global f
    (xx)=f(mu,k)
    errv[k]=norm(myproblem.Afun(mu)*xx-myproblem.b)/norm(myproblem.b)
end
its=range(1,stop=kk,length=kk);
r1 = 1.880361604272027; #This is taken from TDS Arnoldi
err_ref = ((norm(mu)*r1).^(its));

figure(1)
ax = PyPlot.gca()
ax.set_yscale("log")
semilogy(its,errv,"o")
semilogy(its,err_ref)
xlabel("its")
ylabel("res")
legend(["Observed","Predicted"])
