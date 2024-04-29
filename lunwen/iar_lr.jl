using LinearAlgebra
"""
    (V,H,hist)=iar_lr(prob,startvec,N)
    
### Low-rank Infinite Arnoldi
Low-rank version of Infinite Arnoldi method as described in Algorithm 2 (Taylor coefficients) in "A rank-exploiting infinite Arnoldi algorithm for nonlinear eigenvalue problems", R. Van Beeumen, E. Jarlebring, and W. Michiels

Runs `N` steps of the infinite Arnoldi method started with `startvec` with target `σ=0`, specifically for the low-rank problem.
The argument `prob` represents a problem `M(λ)` and has associated functions
`compute_Mder`, `compute_Mlincomb`:

* `M=compute_Mder(prob,s)` computes the matrix function value at `s=λ`.
* `z=compute_Mlincomb(prob,s,X)` computes a linear combination of derivatives. The linear combination coefficients are given in the matrix `X`:  `z=M(s)*X[:,1]+M'(s)*X[:,2]+...+M^{(k)}(s)*X[:,k+1]`.


"""
function iar_lr(prob,startvec,N,precomputed_factorization=nothing)


    T=ComplexF64; # Problem type

    n=length(startvec);

    sigma=0;  # Only target implemented

    s=prob.s
    p=size(prob.V,2)
    @show s,p


    x0=zeros(T,n);
    x0[:]=normalize(startvec);
    H=zeros(T,N+1,N);
    print("IAR L-R Iteration:");
    # Assume N>s+1
    V=zeros(T,n*s+p*(N-s+1),N+1);
    V[1:n,1]=x0;


    start_time=time();
    hist=Dict(:time_count => NaN*zeros(N))



    if (isnothing(precomputed_factorization))
        Msolve=factorize(compute_Mder(prob,sigma));
    else
        Msolve=precomputed_factorization;
    end


    for k=1:N
        print("$k ")

        # Sizes of the blocks in V
        # V:
        # n n n n n n  ^
        #   n n n n n  |
        #     n n n n  s
        #       ..     |
        #       n   n  _
        #        p  p
        #
        #



        # X1 in C^nx(s-1)  (independent of k)
        vecX1=V[1:(s-1)*n,k]
        X1=reshape(vecX1,n,s-1);

        # X2 in C^n  (independent of k)
        X2=V[(s-1)*n .+ (1:n),k]

        # X3 in C^p x (k-s)
        vecX3=V[(s*n+1):(s*n+p*(k-s)),k]
        X3=reshape(vecX3,p,max(k-s,0))

        # x[1]=X1[:,1]   in C^n
        # x[2]=X1[:,2]   in C^n
        # ..
        # x[s-s]=X1[:,s-1]   in C^n  (X2)
        # x[s]=X2  in C^n  (X2)
        # x[s+1]=X3[:,1] in C^p
        # x[s+2]=X3[:,2] in C^p
        # ...
        x=Vector{Vector{T}}(undef,max(k,s))
        for i=1:s-1
            # compute C^n vector
            x[i]=X1[:,i];
        end
        x[s]=X2;
        for i=(s+1):k
            # compute C^p vector
            x[i]=X3[:,i-s];
        end






        y0=zeros(T,n);
        for i=1:s
            y0 += (prob.Aders[i+1]*x[i])/i
        end
        for i=(s+1):k
            y0 += (prob.Uv[i+1]*x[i])/i
        end


        xtilde=-(Msolve\y0)

        # In manuscript: D_{s-1,s-1}
        Dsm1=Diagonal( 1 ./ (1:(s-1)));
        # In manuscript: Dtile_{s+1,m}
        Dtilde=Diagonal( 1 ./ ((s+1):k))



        vecY=[xtilde;vec(X1*Dsm1); (1/s)*prob.V'*X2 ; vec(X3*Dtilde)]
        #vecY=[xtilde;vec(Y1)];

        # Orthogonalization

        if (k<(s+1))
            NN=n*k
        else
            NN=n*s+p*(k-s)
        end
        NN1=size(vecY,1);
        #@show NN,NN1
        V0=V[1:NN,1:k];
        h=V0'*vecY[1:NN];
        vecY=vecY-V[1:NN1,1:k]*h;

        g=V0'*vecY[1:NN];   # twice to be sure
        vecY=vecY-V[1:NN1,1:k]*g;
        h=h+g;

        beta=norm(vecY);  # normalize
        vecY=vecY/beta;
        H[1:(k+1),k]=[h;beta];
        V[1:NN1,k+1]=vecY;

        hist[:time_count][k]=time()-start_time;

    end


    println("done");

    return (V,H,hist);

end
