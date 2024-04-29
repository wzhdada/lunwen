include("iar.jl");
include("iar_lr.jl");
include("tiar.jl");

function infgmres(prob,m;method=:iar)
    n=size(prob.b,1);
    σ=0.0;
    start_time=time();
    println("Infgmres version: $method");
    println("Precomputing");
    Afact=factorize(compute_Mder(prob,σ));

    btilde=-(Afact\prob.b);
    precomp_time=time()-start_time;

    if (method == :iar)
        (Q,H,hist)=iar(prob,btilde,m,Afact);
    elseif (method == :iar_lr)
        (Q,H,hist)=iar_lr(prob,btilde,m,Afact);
    elseif (method == :tiar)
        (Z,A_tens,H,hist)=tiar(Float64,prob,btilde,m,Afact)
        Q = zeros(n,m);
        for j=1:m
            for l=1:m
                Q[1:n,j] += A_tens[1,j,l]*Z[:,l];
            end
        end
    else
        error("Unknown method $method");
    end


    hist[:precomp_time]=[precomp_time];


    e = zeros(m+1,1); e[1] = 1;

    solver=(mu,l) -> Q[1:n,1:l]*((mu*H[1:(l+1),1:l]-Matrix{Float64}(I,l+1,l))\e[1:(l+1)]*norm(btilde));
    return (solver,hist)

end
