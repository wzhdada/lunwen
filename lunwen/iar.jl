using LinearAlgebra
"""
        (V,H,hist)=iar(prob,startvec,N)
        
### Infinite Arnoldi method
Infinite Arnoldi method, as described in Algorithm 2 in  "A linear eigenvalue algorithm for the nonlinear eigenvalue problem",
    by Jarlebring, Elias and Michiels, Wim and Meerbergen, Karl.

Runs `N` steps of the infinite Arnoldi method started with `startvec` with target `σ=0`.
The argument `prob` represents a problem `M(λ)` and has associated functions
`compute_Mder`, `compute_Mlincomb`:

* `M=compute_Mder(prob,s)` computes the matrix function value at `s=λ`.
* `z=compute_Mlincomb(prob,s,X)` computes a linear combination of derivatives. The linear combination coefficients are given in the matrix `X`:  `z=M(s)*X[:,1]+M'(s)*X[:,2]+...+M^{(k)}(s)*X[:,k+1]`.


"""
function iar(prob,startvec,N,precomputed_factorization=nothing)


    T=ComplexF64; # Problem type

    n=length(startvec);

    sigma=0;  # Only target implemented

    x0=zeros(T,n);
    x0[:]=normalize(startvec);
    H=zeros(T,N+1,N);
    print("IAR Iteration:");
    V=zeros(T,n*(N+1),N+1);
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
        # Create L matrix: Equation (17) in nummath paper
        L=Diagonal(1 ./(1:k));


        vecx=V[1:(k*n),k];
        X=reshape(vecx,n,k);


        # Compute y1,..yn: Equation (13) in nummath paper
        Y=zeros(T,size(X,1),size(X,2)+1);
        Y[:,2:end]=X*L;


        # Compute y0: Equation (18) in nummath paper
        # y0=nep.M_lin_comb(X,Y,sigma);
        if (true)
            # Note: First col of Y is zero
            z0=compute_Mlincomb(prob,sigma,Y);
            #z0=Mlinprob.M_lin_comb(Y(:,2:end),1);
        else
            z0=compute_Mlincomb(prob,sigma,X[:,1:k]);
        end

        y0=-(Msolve\z0);
        Y[:,1]=y0;
        vecY=reshape(Y,n*(k+1),1);



        if (true)
            kk=k+1;
        else
            kk=1; # short vector orthogonalization
        end
        V0=V[1:(kk*n),1:k];
        h=V0'*vecY[1:(kk*n)];
        vecY=vecY-V[1:((k+1)*n),1:k]*h;

        g=V0'*vecY[1:(kk*n)];   # twice to be sure
        vecY=vecY-V[1:((k+1)*n),1:k]*g;
        h=h+g;

        beta=norm(vecY);  # normalize
        vecY=vecY/beta;
        H[1:(k+1),k]=[h;beta];
        V[1:((k+1)*n),k+1]=vecY;

        hist[:time_count][k]=time()-start_time;

    end


    println("done");

    return (V,H,hist);

end
