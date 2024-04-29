# Figures 5.4a and 5.4b "Infinite GMRES for parameterized linear systems"
using PyPlot
pygui(true)

a=0.;
b=1.;

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
        return k0;
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

L0=b; n=100;
xs1 = range(0,stop=L0,length=n);
ys1 = k_eval2.(xs1);
ys2 = h_eval2.(xs1);

figure(1)
ax = PyPlot.gca()
xlabel("x")
ylabel("k(x)")
plot(xs1,ys1)

figure(2)
ax = PyPlot.gca()
xlabel("x")
ylabel("h(x)")
plot(xs1,ys2)
