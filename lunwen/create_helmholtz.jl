
using SparseArrays
using LinearAlgebra

#
import Base.exp;
import Base.sin;
import Base.cos;
function exp(A::Matrix{Complex{BigFloat}})
    Z=zero(A);
    n=size(A,1);
    Ak=Matrix{eltype(A)}(I,n,n)
    for k=0:100
        Z+= Ak/(factorial(big(k)));
        Ak[:] =Ak*A;
    end

    return Z
end
function sin(A::Matrix{BigFloat})
    return (exp(1im*A)-exp(-1im*A))/2im
end

function cos(A::Matrix{BigFloat})
    return (exp(1im*A)+exp(-1im*A))/2
end

struct HelmholtzProblem
    sigma
    Aders
    Uv
    V
    Afun
    b
    #
    dx
    xv
    s
end

function compute_Mlincomb(prob::HelmholtzProblem,sigma,Y)
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
function compute_Mder(prob::HelmholtzProblem,mu)
    return prob.Afun(mu);
end


"""

    prob=create_helmholtz(n,a,b,c,N,k_eval,beta_eval,h_eval)

Creates an object representin the discretization of

```u''(x)+(1+mu*k(x))^2*u+beta(x)*u=h(x)```

by discretizing the interval [a,b] with n discretization
points. In the domain [b,c] we have

* `beta(x)=0`
* `k(x)=k0=k(c)`.

"""
function create_helmholtz(n,a,b,c,N,k_eval,beta_eval,h_eval)

    k0=k_eval(c);

    s=2;

    # Discretize interior interval [a,b]
    xv=Vector(range(a,b,length=n+1))[2:end];
    dx=xv[2]-xv[1];

    # Second derivative
    DD=spdiagm(0 => -2*ones(n-1),
               1 => ones(n-1),
               -1=> ones(n-2))/(dx^2)

    # Create auxiliary diagonal matrices
    K_eval=spdiagm(0=>[k_eval.(xv[1:end-1]);0]);
    L=spdiagm(0=>[beta_eval.(xv[1:end-1]);0]);
    II= spdiagm(0=>[ones(n-1);0]);
    K=mu -> spdiagm(0=>[(1 .+ mu*k_eval.(xv[1:end-1])).^2;0]);

    # Boundary condition matrices
    Xn=spzeros(n,2); Xn[n,1]=1; Xn[n,2]=1;
    Yn=spzeros(n,2);
    Yn[n,1]=1;
    Yn[n,2]=3/(2*dx);
    Yn[n-1,2]=-2/dx;
    Yn[n-2,2]=1/(2*dx);

    g=mu->cos((c-b)*(I+mu*k0))
    f=mu->(I+mu*k0)\sin((c-b)*(I+mu*k0));
    F=mu->spdiagm(0=>[g(mu);f(mu)]);


    # Create the A-function

    Afun1=mu->DD+K(mu)+L;
    Afun2=mu->Xn*F(mu)*Yn';
    Afun=mu->Afun1(mu)+Afun2(mu);


    # Compute derivatives of A
    ders=Vector{typeof(DD)}(undef,N);

    ders[1]=DD+II+L;
    ders[2]=2*K_eval;
    ders[3]=2*K_eval*K_eval;


    # Compute derivatives of boundary conditions
    J=big.(diagm(0=>zeros(N), 1=>ones(N-1)));
    fders=Float64.(f(J)[1,:].*factorial.(big.(0:N-1)))
    gders=Float64.(g(J)[1,:].*factorial.(big.(0:N-1)))
    for i=1:N
        Q=Xn*spdiagm(0=>[gders[i];fders[i]])*Yn';
        if (isassigned(ders,i))
            ders[i] += Q;
        else
            ders[i]=Q;
        end
    end



    rhs=[h_eval.(xv[1:end-1]);0] # right-hand side

    #
    Uv_type=Matrix{eltype(DD)}
    Uv=Vector{Uv_type}(undef,N);


    for i=(s+2):N
        Uv[i]=Xn*spdiagm(0=>[gders[i];fders[i]]);
    end
    V=Yn;

    myproblem=HelmholtzProblem(0,ders,Uv,V,Afun,rhs,dx,xv,s);

end
