# Figures 6.3c and 6.3d in "Infinite GMRES for parameterized linear systems"
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
mu=3;
kk=N-1;

# Discretize the problem
myproblem=create_helmholtz2D(N,A0,A1,A2,A3,A4,b);

how_many = 5;
timestamps_mat=zeros(kk,how_many); timestamps=zeros(kk);
for i=1:how_many
    global ff, hist1, timestamps_mat, timestamps
    (ff,hist1)=infgmres(myproblem,kk,method=:tiar);
    timestamps=hist1[:time_count][1:kk] .+ hist1[:precomp_time]
    timestamps_mat[:,i]=timestamps;
end
for i=1:kk
    timestamps[i]=float(median(timestamps_mat[i,:]))
end

errv=zeros(kk);
for k=1:kk-1
    xx=ff(mu,k)
    errv[k]=norm(myproblem.Afun(mu)*xx-b)/norm(b)
end
its=range(1,stop=kk,length=kk);

figure(1)
ax = PyPlot.gca()
ax.set_yscale("log")
xlabel("its")
ylabel("res")
semilogy(its,errv,"o")

figure(2)
ax = PyPlot.gca()
ax.set_yscale("log")
xlabel("CPU-time")
ylabel("res")
semilogy(timestamps,errv,"o")
