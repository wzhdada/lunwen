# Copied from NEP-PACK

using IterativeSolvers
using LinearAlgebra
using Random

"""
   (Z,a,H,hist)=tiar(Float64,prob,btilde,m,Afact)

#from: https://github.com/nep-pack/NonlinearEigenproblems.jl/blob/master/src/method_tiar.jl

tiar(nep,[maxit=30,][σ=0,][γ=1,][linsolvecreator=DefaultLinSolverCreator(),][tolerance=eps()*10000,][neigs=6,][errmeasure,][v=rand(size(nep,1),1),][logger=0,][check_error_every=1,][orthmethod=DGKS,][proj_solve=false,][inner_solver_method=DefaultInnerSolver(),][inner_logger=0])

Run the tensor infinite Arnoldi method on the nonlinear eigenvalue problem stored in `nep`. This is equivalent to `iar`, but handles orthogonalization with
a tensor representation.

The target `σ` is the center around which eiganvalues are computed.
The value `γ` corresponds
to scaling and specifying a shift and scaling is effectively the same as the
transformation `λ=γs+σ` where `s`
is now the eigenvalue parameter. If you want eigenvalues in a disk centered,
select `σ` as the center
of the disk and `γ` as the radius.
The vector `v` is the starting vector for constructing the Krylov space. The orthogonalization
method, used in contructing the orthogonal basis of the Krylov space, is specified by
`orthmethod`, see the package `IterativeSolvers.jl`.
The iteration is continued until `neigs` Ritz pairs have converged.
This function throws a `NoConvergenceException` if the wanted eigenpairs are not computed after `maxit` iterations.
However, if `neigs` is set to `Inf` the iteration is continued until `maxit` iterations without an error being thrown.
The parameter `proj_solve` determines if the Ritz paris are extracted using the Hessenberg matrix (false),
or as the solution to a projected problem (true). If true, the method is descided by `inner_solver_method`, and the
logging of the inner solvers are descided by `inner_logger`, which works in the same way as `logger`.

See [`augnewton`](@ref) for other parameters.

### References
Algorithm 2 in Jarlebring, Mele, Runborg, The Waveguide Eigenvalue Problem and the Tensor Infinite Arnoldi Method



"""
function tiar(
    ::Type{T},
    nep,
    b,maxit,precomputed_factorization=nothing;
    orthmethod::Type{T_orth}=DGKS,
    tol=eps(real(T))*10000,
    σ=zero(T),
    γ=one(T),
    check_error_every=1)  where{T,T_orth<:IterativeSolvers.OrthogonalizationMethod}

    start_time=time();
    hist=Dict(:time_count => NaN*zeros(N))


    if (isnothing(precomputed_factorization))
        Msolve=factorize(compute_Mder(nep,σ));
    else
        Msolve=precomputed_factorization;
    end

    # Ensure types σ and v are of type T
    σ=T(σ)
    v=Array{T,1}(b)

    # initialization
    n = size(b,1); m = maxit;

    # initialize variables
    a  = zeros(T,m+1,m+1,m+1);
    Z  = zeros(T,n,m+1);
    t  = zeros(T,m+1);
    tt = zeros(T,m+1);
    g  = zeros(T,m+1,m+1);
    f  = zeros(T,m+1,m+1);
    ff = zeros(T,m+1,m+1);
    H  = zeros(T,m+1,m);
    h  = zeros(T,m+1);
    hh = zeros(T,m+1);
    y  = zeros(T,n,m+1);
    α=Array{T,1}(γ.^(0:m)); α[1]=zero(T);

    # local M0inv::LinSolver=create_linsolver(linsolvercreator,nep,σ)


    err = NaN*ones(m+1,m+1);
    λ=zeros(T,m+1); Q=zeros(T,n,m+1);
    Z[:,1]=v; Z[:,1]=Z[:,1]/norm(Z[:,1]);
    a[1,1,1]=one(T);

    print("TIAR Iteration:");

    k=1; conv_eig=0;
    for k=1:m
        print("$k ")

        # computation of y[:,2], ..., y[:,k+1]
        y[:,2:k+1]=Z[:,1:k]*transpose(a[1:k,k,1:k])
        broadcast!(/,view(y,:,2:k+1),view(y,:,2:k+1),(1:k)')

        # computation of y[:,1]
        YY=y[:,1:k+1]*Diagonal(α[1:k+1])
        y[:,1] = compute_Mlincomb(nep,σ,YY);
        y[:,1] = -(Msolve\y[:,1]);

        # Gram–Schmidt orthogonalization in Z
        Z[:,k+1]=y[:,1];
        t[k+1] = orthogonalize_and_normalize!(view(Z,:,1:k), view(Z,:,k+1), view(t,1:k), orthmethod)

        # compute the matrix G
        for l=1:k+1
            for i=2:k+1
                g[i,l]=a[i-1,k,l]/(i-1);
            end
         g[1,l]=t[l];
        end

        # compute h (orthogonalization with tensors factorization)
        h = zero(h)
        Ag = zero(h[1:k])
        for l=1:k
            mul!(Ag,a[1:k,1:k,l]',g[1:k,l])
            h[1:k] .+= Ag;
        end

        # compute the matrix F
        f=g;
        Ah = zero(f[1:k+1,1])
        for l=1:k
            mul!(Ah,a[1:k+1,1:k,l],h[1:k])
            f[1:k+1,l] .-= Ah;
        end

        # re-orthogonalization
        # compute hh (re-orthogonalization with tensors factorization)
        hh = zero(hh)
        Af = zero(hh[1:k])
        for l=1:k
            mul!(Af,a[1:k,1:k,l]',f[1:k,l])
            hh[1:k] .+= Af;
        end

        # compute the matrix FF
        ff=f;
        Ah=zero(ff[1:k+1,1])
        for l=1:k
            mul!(Ah,a[1:k+1,1:k,l],hh[1:k])
            ff[1:k+1,l] .-= Ah;
        end

        # update the orthogonalization coefficients
        h=h+hh; f=ff;
        β=norm(view(f,1:k+1,1:k+1)); # equivalent to Frobenius norm

        # extend the matrix H
        H[1:k,k]=h[1:k]; H[k+1,k]=β;

        # extend the tensor
        for i=1:k+1
            for l=1:k+1
                a[i,k+1,l]=f[i,l]/β;
            end
        end
        hist[:time_count][k]=time()-start_time;

    end
    return (Z,a,H,hist)
end
