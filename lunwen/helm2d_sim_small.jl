# Figures 6.3a and 6.3b in "Infinite GMRES for parameterized linear systems"

using SparseArrays
using LinearAlgebra
using PyPlot
using Statistics
pygui(true)
include("infgmres.jl");
include("create_helmholtz_2d.jl");

data=matread("remove_mu.mat")
A0=data["A0"]; A1=data["A1"];
A2=data["A2"]; A3=data["A3"];
A4=data["A3"]; b=vec(data["bvec_2d"]);

# Number of derivatives
N=51;
mu=.1;
kk=10;

# Discretize the problem
myproblem=create_helmholtz2D(N,A0,A1,A2,A3,A4,b);

how_many = 5;
timestamps_mat=zeros(kk,how_many); timestamps=zeros(kk);
timestamps_mat2=zeros(kk,how_many); timestamps2=zeros(kk);
for i=1:how_many
    global ff, hist1, timestamps
    global ff2, hist2, timestamps2
    (ff,hist1)=infgmres(myproblem,kk,method=:tiar);
    (ff2,hist2)=infgmres(myproblem,kk,method=:iar);
    timestamps=hist1[:time_count][1:kk] .+ hist1[:precomp_time]
    timestamps_mat[:,i]=timestamps;
    timestamps2=hist2[:time_count][1:kk] .+ hist2[:precomp_time]
    timestamps_mat2[:,i]=timestamps2;
end
for i=1:kk
    timestamps[i]=float(median(timestamps_mat[i,:]))
    timestamps2[i]=float(median(timestamps_mat2[i,:]))
end

errv=zeros(kk);
errv2=zeros(kk);
for k=1:kk-1
    xx=ff(mu,k)
    xx2=ff2(mu,k)
    errv[k]=norm(myproblem.Afun(mu)*xx-b)/norm(b)
    errv2[k]=norm(myproblem.Afun(mu)*xx2-b)/norm(b)
end
its=range(1,stop=kk,length=kk);


figure(1)
ax = PyPlot.gca()
ax.set_yscale("log")
xlabel("its")
ylabel("res")
semilogy(its,errv,"o")
semilogy(its,errv2,"+")
legend(["Tensor","Alg 1"])

figure(2)
ax = PyPlot.gca()
ax.set_yscale("log")
xlabel("CPU-time")
ylabel("res")
semilogy(timestamps,errv,"o")
semilogy(timestamps2,errv2,"+")
legend(["Tensor","Alg 1"])
