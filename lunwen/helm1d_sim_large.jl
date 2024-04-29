# Figures 5.5b, 5.7a and 5.7b in "Infinite GMRES for parameterized linear systems"
using SparseArrays
using LinearAlgebra
using PyPlot
using Statistics
pygui(true)
include("infgmres.jl");
include("create_helmholtz.jl");

n=5000;
a=0;
b=1;
c=1.5;


function envelope_k(x)
    L0=b;
    alpha=10;
    if (x<L0/2)
        return 5+10*(x/L0)*sin((1/L0)*alpha*pi*x);
    else
        return 5+10*(1-x/L0)*sin((1/L0)*alpha*pi*x);
    end
end

function k_eval2(x)
    k0 = envelope_k(b);
    if (x<b)
        return envelope_k(x) # k_interior
    else
        return k0; # Never used?
    end

end

function beta_eval2(x)
    if (x<b)
        return sin((x-a)*2*pi/(b-a));
    else
        return 0;
    end

end

function h_eval2(x)
    if x<b
        return ((x-b)^2)/(a-b)^2;
    else
        return 0;
    end
end


# Number of derivatives
N=43;
myproblem=create_helmholtz(n,a,b,c,N,k_eval2,beta_eval2,h_eval2);

kk=N-1;
how_many = 5;

timestamps_mat1=zeros(kk,how_many); timestamps1=zeros(kk);
timestamps_mat2=zeros(kk,how_many); timestamps2=zeros(kk);
timestamps_mat3=zeros(kk,how_many); timestamps3=zeros(kk);
for i=1:how_many
    global ff1, hist1
    global ff2, hist2
    global ff3, hist3
    (ff1,hist1)=infgmres(myproblem,kk,method=:tiar);
    (ff2,hist2)=infgmres(myproblem,kk,method=:iar);
    (ff3,hist3)=infgmres(myproblem,kk,method=:iar_lr);
    timestamps1a=hist1[:time_count][1:kk] .+ hist1[:precomp_time]
    timestamps_mat1[:,i]=timestamps1a;
    timestamps2a=hist2[:time_count][1:kk] .+ hist2[:precomp_time]
    timestamps_mat2[:,i]=timestamps2a;
    timestamps3a=hist3[:time_count][1:kk] .+ hist3[:precomp_time]
    timestamps_mat3[:,i]=timestamps3a;
end
for i=1:kk
    timestamps1[i]=float(median(timestamps_mat1[i,:]))
    timestamps2[i]=float(median(timestamps_mat2[i,:]))
    timestamps3[i]=float(median(timestamps_mat3[i,:]))
end


mmu=2.5;

errv1=zeros(kk); errv2=zeros(kk); errv3=zeros(kk);
for k=1:kk
    global xx1, xx2, xx3
    xx1=ff1(mmu,k)
    xx2=ff2(mmu,k)
    xx3=ff3(mmu,k)
    errv1[k]=norm(myproblem.Afun(mmu)*xx1-myproblem.b)/norm(myproblem.b)
    errv2[k]=norm(myproblem.Afun(mmu)*xx2-myproblem.b)/norm(myproblem.b)
    errv3[k]=norm(myproblem.Afun(mmu)*xx3-myproblem.b)/norm(myproblem.b)
end
its=range(1,stop=kk,length=kk);

figure(1)
ax = PyPlot.gca()
#plot(myproblem.xv,real(xx1))
plot(myproblem.xv,real(xx2))
xlabel("xs")
ylabel("u_n(x)")
#plot(myproblem.xv,real(xx3))

figure(2)
ax = PyPlot.gca()
ax.set_yscale("log")
semilogy(its,errv1,"o")
semilogy(its,errv2,"+")
semilogy(its,errv3,"*")
xlabel("its")
ylabel("res")
ax.legend(["Tensor","Alg 1","Low-rank"])

figure(3)
ax = PyPlot.gca()
ax.set_yscale("log")
semilogy(timestamps1,errv1,"o")
semilogy(timestamps2,errv2,"+")
semilogy(timestamps3,errv3,"*")
xlabel("CPU-time")
ylabel("res")
ax.legend(["Tensor","Alg 1","Low-rank"])
