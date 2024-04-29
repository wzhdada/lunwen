
using SparseArrays
using LinearAlgebra
using MAT
import Base.exp;
import Base.sin;
import Base.cos;

struct HelmholtzProblem2D
    sigma
    Aders
    Afun
    b
end

struct HelmholtzProblem2D_nlterms
    sigma
    Afun
    fv
    fders
    Cv
    b
end

function compute_Mlincomb(prob::HelmholtzProblem2D,sigma,Y)
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
function compute_Mder(prob::HelmholtzProblem2D,mu)
    return prob.Afun(mu);
end


"""

    prob=create_helmholtz(n,a,b,c,N,k_eval,beta_eval,h_eval)

Creates an object representing the discretization of

```u''(x)+f1(mu)*(1+mu*k(x))^2*u+f2(mu)*beta(x)*u=h(x)```

using matrices from a finite element discretization.

"""
function create_helmholtz2D(N,A0,A1,A2,A3,A4,b)

    n = length(b);

    ders=Vector{typeof(A0)}(undef,N);
    ders[1] = A0;
    ders[2] = A1 + A4;


    coeff=1;
    for i=4:2:N
       coeff = coeff*(-1);
       ders[i] = coeff*A4;
    end
    ders[4] += 6*A3;
    for i=3:2:N
        ders[i] = spzeros(n,n);
    end
    ders[3] = 4*A2;

    # Create the A-function
    Afun = mu-> (A0 + mu*A1 + (2*mu^2)*A2 + (mu^3)*A3 + sin(mu)*A4);

    rhs=b;  # right-hand side

    myproblem=HelmholtzProblem2D(0,ders,Afun,rhs);

end
