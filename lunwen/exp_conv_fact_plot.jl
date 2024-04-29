# Figure 5.3b in "Infinite GMRES for parameterized linear systems"
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
(f,hist1)=infgmres(myproblem,30,method=:iar);
max_it = 20;
v = range(-5,stop=-2,length=50);
mu_list = [10^i for i in v];
tol1 = 1.e-10;
obs = zeros(length(mu_list),1);
pred1 = zeros(length(mu_list),1);
r1 = 9.304668750449350; #This is taken from TDS Arnoldi

for j = 1:length(mu_list)
    global mu
    res = zeros(max_it,1);
    mu = mu_list[j];
    A_of_mu = myproblem.Afun(mu);

    for m=1:max_it
        global f
        (xx)=f(mu,m)
        res[m] = norm(A_of_mu*xx - myproblem.b)/norm(myproblem.b);
    end

    #Take the worst reducation in the residual
    ratio_list=zeros(max_it,1);
    end_m=0;
    for i=1:length(res)
        if res[i] < tol1
            end_m = i;
            break;
        end
    end
    for i=2:end_m
        ratio_list[i-1]=res[i]/res[i-1];
    end
    obs[j]=maximum(ratio_list);
    pred1[j]=mu*r1;

end

figure(1)
ax = PyPlot.gca()
ax.set_yscale("log")
ax.set_xscale("log")
xlabel("mu")
ylabel("rho")
plot(mu_list,obs,"o")
plot(mu_list,pred1)
legend(["Observed","Predicted"])
