using CUDA

# z-> CuDeviceArray{F, 5, 1} : 5 dimensioni
#     - 1°dim : parte reale e immaginaria 
#     - 2°dim : ordine perturbativo 
#     - 3°dim : coordinata temporale del sito 
#     - 4°dim : coordinata spaziale del sito 
#     - 5°dim : componente del vettore 
# U-> CuDeviceArray{F, 5, 1} : 5 dimensioni
#     - 1°dim : parte reale e immaginaria (I(1)=Re e I(2)=Im)
#     - 2°dim : ordine perturbativo ( 1=ordine zero )
#     - 3°dim : coordinata temporale del sito 
#     - 4°dim : coordinata spaziale del sito 
#     - 5°dim : direzione lungo il reticolo ( mu = 1 tempo (sx-dx) e mu = 2 spazio (up-dw) )

function create_kernels(::Type{F}, Npoint::I, max_ptord::I, n_comps::I, n_copies::I) where {F <: AbstractFloat, I <: Integer}

    n_ptords = max_ptord + one(I) # total number of perturbative orders
    N_colors = n_comps + one(n_comps) #total number of colors 

    # Functions returning the coordinates of the nearest neighbour of site (i, k) in the given direction
    # Are there periodic boundary conditions?? 
    @inline function neighbour_up(i::I, k::I)
        return (mod(i-I(2), Npoint) + one(I), k)
    end

    @inline function neighbour_down(i::I, k::I)
        return (mod(i, Npoint) + one(I), k)
    end

    @inline function neighbour_left(i::I, k::I)
        return (i, mod(k-I(2), Npoint) + one(I))
    end

    @inline function neighbour_right(i::I, k::I)
        return (i, mod(k, Npoint) + one(I))
    end

    @inline function get_indexes()
        # converts from one-dimensional index ind to two-dimensional indexes i, k

        ind = (blockIdx().x - one(I)) * blockDim().x + threadIdx().x
        i::I = rem(ind-one(I), Npoint) + one(I)
        k::I = div(ind-one(I), Npoint) + one(I)

        #assign an index to the copies 
        c::I = (blockIdx().y - one(I)) * blockDim().y + threadIdx().y
        return (i, k, c)
    end
    
    @inline function mult_sc_diffsites!(
        a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 4, 1}, result::CuDeviceArray{F, 4, 1}, i_a::I, k_a::I,
        i_b::I, k_b::I, i_out::I, k_out::I, c::I
    )
        # questa moltiplicazione è pensata per moltiplicare dei moduli quadri
        # sono ancora oggetti perturbativi con la struttura di reticolo
        # ma sono scalari quindi non c'è l'indice di componente
        # reset_sc!(result, i, k)
        for j::I = 1:n_ptords
            @inbounds result[j, i_out, k_out, c] = zero(F)
            for l::I = 1:j
                @inbounds result[j, i_out, k_out, c] += a[l, i_a, k_a, c] * b[j-l+one(I), i_b, k_b, c]
            end
        end
        return
    end 

    @inline function mult_complvec_by_complscal!(a::CuDeviceArray{F, 6, 1}, b::CuDeviceArray{F, 5, 1}, result::CuDeviceArray{F, 6, 1},
        i_out::I, k_out::I, i_a::I, k_a::I, i_b::I, k_b::I, c::I
    )

        # se non inizializzo il result a zero posso passare direttamente il vettore gradiente in result
        for n::I = 1:n_comps 
            for j::I = 1:n_ptords 
                for l::I = 1:j 
                    @inbounds a_re = a[I(1), l, i_a, k_a, n, c]
                    @inbounds a_im = a[I(2), l, i_a, k_a, n, c]
                    @inbounds b_re = b[I(1), j-l+I(1), i_b, k_b, c]
                    @inbounds b_im = b[I(2), j-l+I(1), i_b, k_b, c]

                    @inbounds result[I(1), j, i_out, k_out, n, c] += (a_re * b_re - a_im * b_im)
                    @inbounds result[I(2), j, i_out, k_out, n, c] += (a_im * b_re + a_re * b_im)
                end 

            end 
        end 

        return 
    end 

    @inline function mult_compvec_by_U!(z::CuDeviceArray{F, 6, 1}, U::CuDeviceArray{F, 6, 1}, 
        result::CuDeviceArray{F, 6 ,1}, i_a::I, k_a::I, i_b::I, k_b::I, 
        i_out::I, k_out::I, mu::I, c::I
    )
        # Multiplies a complex vector by U_mu
        for n::I = 1:n_comps  
            for j::I = 1:n_ptords
                for l::I = 1:j 
                    @inbounds z_re = z[I(1), l, i_a, k_a, n, c]
                    @inbounds z_im = z[I(2), l, i_a, k_a, n, c]
                    @inbounds u_re = U[I(1), j-l+I(1), i_b, k_b, mu, c]
                    @inbounds u_im = U[I(2), j-l+I(1), i_b, k_b, mu, c]

                    @inbounds result[I(1), j, i_out, k_out, n, c] += (z_re * u_re - z_im * u_im)
                    @inbounds result[I(2), j, i_out, k_out, n, c] += (z_re * u_im + z_im * u_re)
                end 
            end 
        end 
        return 
    end 

    @inline function mult_compvec_by_Uconj!(z::CuDeviceArray{F, 6, 1}, U::CuDeviceArray{F, 6, 1}, 
        result::CuDeviceArray{F, 6, 1}, i_a::I, k_a::I, i_b::I, k_b::I, 
        i_out::I, k_out::I, mu::I, c::I
    )   
        # Multiplies a complex vector by Uconj_mu
        for n::I = 1:n_comps  
            for j::I = 1:n_ptords
                for l::I = 1:j 
                    @inbounds z_re = z[I(1), l, i_a, k_a, n, c]
                    @inbounds z_im = z[I(2), l, i_a, k_a, n, c]
                    @inbounds uconj_re = U[I(1), j-l+I(1), i_b, k_b, mu, c]
                    @inbounds uconj_im = -U[I(2), j-l+I(1), i_b, k_b, mu, c]

                    @inbounds result[I(1), j, i_out, k_out, n, c] += (z_re * uconj_re - z_im * uconj_im)
                    @inbounds result[I(2), j, i_out, k_out, n, c] += (z_im * uconj_re + z_re * uconj_im)
                end 
            end 
        end 
        return 
    end 

    @inline function complex_exponential_taylor!(
        X::CuDeviceArray{F, 5, 1},  #Array reale 
        E::CuDeviceArray{F, 6, 1}, #Risultato: esponenziale complesso 
        i::I, k::I, mu::I, c::I, 
        potenza_corrente::CuDeviceArray{F, 4, 1},
        temp::CuDeviceArray{F, 4, 1}
    )   
        for j::I = 1:n_ptords
            @inbounds E[I(1), j, i, k, mu, c] = zero(F)
            @inbounds E[I(2), j, i, k, mu, c] = zero(F)
            @inbounds potenza_corrente[j, i, k, c] = X[j, i, k, mu, c] 
        end
        @inbounds E[I(1), I(1), i, k, mu, c] = one(F) # Ordine 0 è 1
        C_m = one(F) # Initialize the binomial factor 

        # Il termine di primo ordine per l'esponenziale complesso contribuisce solo alla parte immaginaria 
        for j::I = 1:n_ptords
            @inbounds E[I(2), j, i, k, mu, c] += C_m * potenza_corrente[j, i, k, c] # Ha senso, pensa allo sviluppo della radice  
        end
        
        # Calcolo iterativo per m >= 2
        for m::I = 2:(max_ptord)
            C_m = C_m / F(m) # Correzione 1: dividi esattamente per m
            
            for j::I = 1:n_ptords
                @inbounds temp[j, i, k, c] = zero(F)
                for l::I = 1:j
                    @inbounds temp[j, i, k, c] += potenza_corrente[l, i, k, c] * X[j-l+one(I), i, k, mu, c]
                end
            end
            
            #Mappatura esatta ed efficiente dei segni di i^m
            resto = rem(m, I(4))
            for j::I = 1:n_ptords
                @inbounds p_val = temp[j, i, k, c]
                @inbounds potenza_corrente[j, i, k, c] = p_val
                
                if resto == zero(I)
                    @inbounds E[I(1), j, i, k, mu, c] += C_m * p_val
                elseif resto == one(I)
                    @inbounds E[I(2), j, i, k, mu, c] += C_m * p_val
                elseif resto == I(2) # rem(2,4) = 2
                    @inbounds E[I(1), j, i, k, mu, c] -= C_m * p_val
                else # resto == 3
                    @inbounds E[I(2), j, i, k, mu, c] -= C_m * p_val
                end 
            end
        end

        return 
    end 

    function ExpU_kernel!(
        X::CuDeviceArray{F, 5, 1},
        Exp::CuDeviceArray{F, 6, 1}, 
        U::CuDeviceArray{F, 6, 1}, 
        ExpU::CuDeviceArray{F, 6, 1},
        sc_buffA::CuDeviceArray{F, 4, 1}, 
        sc_buffB::CuDeviceArray{F, 4, 1}
    )

        i, k, c = get_indexes()

        if (i <= Npoint && k <= Npoint && c <= n_copies)

            for mu::I = 1:2 
                complex_exponential_taylor!(X, Exp, i, k, mu, c, sc_buffA, sc_buffB)
     
                for j::I = n_ptords:-1:1 
                    res_Re = zero(F)
                    res_Im = zero(F)
                    for l::I = 1:j 
                        @inbounds Exp_Re = Exp[I(1), l, i, k, mu, c]
                        @inbounds Exp_Im = Exp[I(2), l, i, k, mu, c]
                        @inbounds U_Re = U[I(1), j-l+I(1), i, k, mu, c]
                        @inbounds U_Im = U[I(2), j-l+I(1), i, k, mu, c]

                        res_Re += Exp_Re * U_Re - Exp_Im * U_Im 
                        res_Im += Exp_Im * U_Re + Exp_Re * U_Im
                    end 
                    @inbounds ExpU[I(1), j, i, k, mu, c] = res_Re
                    @inbounds ExpU[I(2), j, i, k, mu, c] = res_Im
                end 
            end
            
        end 
        return nothing 
    end 

    function zscaled2_kernel!(
    z::CuDeviceArray{F, 6, 1}, 
    nu_vacuum::CuDeviceArray{F, 4, 1}, 
    zscaled2::CuDeviceArray{F, 4, 1} # Risultato finale, array 3D!
    )
        i, k, c = get_indexes()

        if (i <= Npoint && k <= Npoint && c <= n_copies)
            # Estrai il valore del vuoto per la componente N-esima
            idx_N = n_comps + I(1)
            @inbounds val_vac = nu_vacuum[I(1), i, k, idx_N]
            val_vac2 = val_vac * val_vac # Modulo quadro del vuoto classico --> quadrato dell'ultima componente di nu!!
        
            # Inizializza a zero l'accumulatore per tutti gli ordini
            for j::I = 1:n_ptords
                @inbounds zscaled2[j, i, k, c] = zero(F)
            end
        
            # Calcola \sum_n |z_n / val_vac|^2 
            for n::I = 1:n_comps
                for j::I = 2:n_ptords
                    for l::I = 1:j
                        # Z normale ad ordine l
                        @inbounds z_Re = z[I(1), l, i, k, n, c]
                        @inbounds z_Im = z[I(2), l, i, k, n, c]
                    
                        # Z coniugato ad ordine j-l+I(1)
                        @inbounds z_conj_Re = z[I(1), j-l+I(1), i, k, n, c]
                        @inbounds z_conj_Im = -z[I(2), j-l+I(1), i, k, n, c]
                    
                        # Moltiplica, estrai la parte reale e dividi per val_vac^2
                        parte_reale = (z_Re * z_conj_Re - z_Im * z_conj_Im) / val_vac2
                    
                        @inbounds zscaled2[j, i, k, c] += parte_reale
                    end
                end
            end
        end
        return nothing
    end

    @inline function radice_taylor!(
        x::CuDeviceArray{F, 4, 1}, i::I, k::I, c::I, 
        risultato::CuDeviceArray{F, 4, 1},
        potenza_corrente::CuDeviceArray{F, 4, 1},
        temp::CuDeviceArray{F, 4, 1}
    )
        # Sviluppo di sqrt(1 - x) = 1 - 1/2 x - 1/8 x^2 - 1/16 x^3 - ...
        for j::I = 1:n_ptords
            @inbounds risultato[j, i, k, c] = zero(F)
            @inbounds potenza_corrente[j, i, k, c] = x[j, i, k, c] # x all'ordine 1
        end
        @inbounds risultato[1, i, k, c] = one(F) # Ordine 0 è 1

        C_m = F(-0.5) # Coefficiente per m=1

        # Aggiungiamo il termine m=1 -> per j=1 non ho problemi, perché zscaled2 è zero a j=1 !!! 
        for j::I = 1:n_ptords
            @inbounds risultato[j, i, k, c] += C_m * potenza_corrente[j, i, k, c] # Ha senso, pensa allo sviluppo della radice  
        end

        # Calcolo iterativo per m >= 2
        for m::I = 2:(max_ptord)
            C_m = C_m * F(m - 1.5) / F(m) # Ricorsione esatta del coefficiente ( è il coefficiente binomiale )
            mult_sc_diffsites!(potenza_corrente, x, temp, i, k, i, k, i, k, c) # x^m = x^(m-1) * x
            for j::I = 1:n_ptords
                @inbounds potenza_corrente[j, i, k, c] = temp[j, i, k, c]
                @inbounds risultato[j, i, k, c] += C_m * potenza_corrente[j, i, k, c]
            end
        end
        return nothing
    end

    @inline function inv_radice_taylor!(
        x::CuDeviceArray{F, 4, 1}, i::I, k::I, c::I, 
        risultato::CuDeviceArray{F, 4, 1},
        potenza_corrente::CuDeviceArray{F, 4, 1},
        temp::CuDeviceArray{F, 4, 1}
    )
        # Sviluppo di 1 / sqrt(1 - x) = 1 + 1/2 x + 3/8 x^2 + 5/16 x^3 + ...
        for j::I = 1:n_ptords
            @inbounds risultato[j, i, k, c] = zero(F)
            @inbounds potenza_corrente[j, i, k, c] = x[j, i, k, c]
        end
        @inbounds risultato[1, i, k, c] = one(F)

        C_m = F(0.5) # Coefficiente per m=1

        # Aggiungiamo il termine m=1
        for j::I = 1:n_ptords
            @inbounds risultato[j, i, k, c] += C_m * potenza_corrente[j, i, k, c]
        end

        # Calcolo iterativo per m >= 2
        for m::I = 2:(max_ptord)
            C_m = C_m * F(m - 0.5) / F(m) # Ricorsione esatta per l'inverso della radice
            mult_sc_diffsites!(potenza_corrente, x, temp, i, k, i, k, i, k, c)
            for j::I = 1:n_ptords
                @inbounds potenza_corrente[j, i, k, c] = temp[j, i, k, c]
                @inbounds risultato[j, i, k, c] += C_m * potenza_corrente[j, i, k, c]
            end
        end
        return nothing
    end
    
    function roots_kernel!(
        zscaled2::CuDeviceArray{F, 4, 1}, 
        root_zscaled2::CuDeviceArray{F, 4, 1}, 
        invroot_zscaled2::CuDeviceArray{F, 4, 1}, 
        sc_buffA::CuDeviceArray{F, 4, 1},
        sc_buffB::CuDeviceArray{F, 4, 1}
    )
        i, k, c = get_indexes()
        
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            # 1. Calcola z^N(x) = sqrt(1 - zscaled2)
            # Usa sc_buffA e sc_buffB come memoria temporanea
            radice_taylor!(zscaled2, i, k, c, root_zscaled2, sc_buffA, sc_buffB)
            
            # Calcola 1 / z^N(x) = 1 / sqrt(1 - zscaled2)
            # Ricicla la stessa memoria temporanea per il secondo calcolo
            inv_radice_taylor!(zscaled2, i, k, c, invroot_zscaled2, sc_buffA, sc_buffB)
        end 
        
        return nothing
    end

    function init_M1_kernel!(M1::CuDeviceArray{F, 7, 1}) 
        
        i, k, c = get_indexes() 
        
        if (i <= Npoint && k <= Npoint && c <= n_copies)

            
            for mu::I = 1:2 # 1 è la direzione temporale, 2 è la direzione spaziale. 
                for n::I = 1:N_colors
                    for j::I = 1:n_ptords
                        @inbounds M1[I(1), j, i, k, mu, n, c] = zero(F)
                        @inbounds M1[I(2), j, i, k, mu, n, c] = zero(F)
                    end
                end
            end
            
            for mu::I = 1:2
                for n::I = 1:N_colors
                    # Calcolo della fase: in Julia n va da 1 a N_colors, nel C era k da 0 a N-1
                    phase = F(2.0 * pi * (n - 1) / N_colors)
                    re_phase = cos(phase)
                    im_phase = sin(phase)
                    
                    if i == Npoint && mu == 1 # Corrisponde a x0 == Lt - 1 e mu == 0 nel C
                        @inbounds M1[I(1), I(1), i, k, mu, n, c] = re_phase
                        @inbounds M1[I(2), I(1), i, k, mu, n, c] = im_phase
                        
                    else # Tutti gli altri siti e direzioni
                        @inbounds M1[I(1), I(1), i, k, mu, n, c] = one(F)
                        @inbounds M1[I(2), I(1), i, k, mu, n, c] = zero(F)

                    end
                end
            end
        end
        return nothing
    end

    @inline function mult_z_by_M!(z::CuDeviceArray{F, 6, 1}, M::CuDeviceArray{F, 7, 1}, 
        result::CuDeviceArray{F, 6, 1}, i_a::I, k_a::I, i_b::I, k_b::I, 
        i_out::I, k_out::I, mu::I, c::I
    )
        for n::I = 1:n_comps  
            for j::I = 1:n_ptords
                @inbounds result[I(1), j, i_out, k_out, n, c] = zero(F)
                @inbounds result[I(2), j, i_out, k_out, n, c] = zero(F)
                for l::I = 1:j 
                    @inbounds z_re = z[I(1), l, i_a, k_a, n, c]
                    @inbounds z_im = z[I(2), l, i_a, k_a, n, c]
                    @inbounds M_re = M[I(1), j-l+I(1), i_b, k_b, mu, n, c]
                    @inbounds M_im = M[I(2), j-l+I(1), i_b, k_b, mu, n, c]

                    @inbounds result[I(1), j, i_out, k_out, n, c] += (z_re * M_re - z_im * M_im)
                    @inbounds result[I(2), j, i_out, k_out, n, c] += (z_im * M_re + z_re * M_im)
                end 
            end 
        end 
        return 
    end 

    @inline function mult_z_by_Mconj!(z::CuDeviceArray{F, 6, 1}, M::CuDeviceArray{F, 7, 1}, 
        result::CuDeviceArray{F, 6, 1}, i_a::I, k_a::I, i_b::I, k_b::I, 
        i_out::I, k_out::I, mu::I, c::I
    )
        for n::I = 1:n_comps # z è di dimensione N_colors-1 = n_comps, mentre M è dimensione N_colors !!!!
            for j::I = 1:n_ptords
                @inbounds result[I(1), j, i_out, k_out, n, c] = zero(F)
                @inbounds result[I(2), j, i_out, k_out, n, c] = zero(F)
                for l::I = 1:j 
                    @inbounds z_re = z[I(1), l, i_a, k_a, n, c]
                    @inbounds z_im = z[I(2), l, i_a, k_a, n, c]
                    @inbounds M_re = M[I(1), j-l+I(1), i_b, k_b, mu, n, c]
                    @inbounds M_im = -M[I(2), j-l+I(1), i_b, k_b, mu, n, c]

                    @inbounds result[I(1), j, i_out, k_out, n, c] += (z_re * M_re - z_im * M_im)
                    @inbounds result[I(2), j, i_out, k_out, n, c] += (z_im * M_re + z_re * M_im)
                end 
            end 
        end 
        return 
    end 

    @inline function fwd_corrective_term!(
        root::CuDeviceArray{F, 4, 1}, inv_root::CuDeviceArray{F, 4, 1}, 
        z::CuDeviceArray{F, 6, 1}, U::CuDeviceArray{F, 6, 1}, result::CuDeviceArray{F, 6, 1}, 
        i::I, k::I, i_fwd::I, k_fwd::I, c::I,     
        nu_vac::CuDeviceArray{F, 4, 1}, M::CuDeviceArray{F, 7, 1}, mu::I, 
        c_sc_buffA::CuDeviceArray{F, 5, 1}, c_sc_buffB::CuDeviceArray{F, 5, 1}, sc_buffA::CuDeviceArray{F, 4, 1}
    )
        
        # Rapporto z^N(x+mu) / z^N(x) 
        # Firma: out(i,k), a(i_bwd, k_bwd), b(i,k)
        mult_sc_diffsites!(root, inv_root, sc_buffA, i_fwd, k_fwd, i, k, i, k, c) # rapporto delle radici 
        
        # Background e Matrice di Twist M(x) [Ordine g^0]
        idx_N = n_comps + I(1)
        @inbounds ratio_vac = nu_vac[I(1), i_fwd, k_fwd, idx_N] / nu_vac[I(1), i, k, idx_N]
        
        @inbounds M_re =  M[I(1), I(1), i, k, mu, idx_N, c] # La matrice è zero a tutti gli ordine superiori a quello banale 
        @inbounds M_im =  -M[I(2), I(1), i, k, mu, idx_N, c] # questi sono gli ultimi elementi della matrice di twist 

        for j::I = 1:n_ptords
            @inbounds val = sc_buffA[j, i, k, c] * ratio_vac
            @inbounds c_sc_buffA[I(1), j, i, k, c] = val * M_re
            @inbounds c_sc_buffA[I(2), j, i, k, c] = val * M_im
        end 

       #Moltiplicazione per U_mu(x)
        for j::I = 1:n_ptords
            @inbounds c_sc_buffB[I(1), j, i, k, c] = zero(F)
            @inbounds c_sc_buffB[I(2), j, i, k, c] = zero(F)
            
            for l::I = 1:j
                # Prefattore calcolato al passo precedente
                @inbounds A_re = c_sc_buffA[I(1), l, i, k, c]
                @inbounds A_im = c_sc_buffA[I(2), l, i, k, c]
                
                # Link di gauge valutato in x 
                @inbounds U_re =  U[I(1), j-l+I(1), i, k, mu, c] 
                @inbounds U_im =  -U[I(2), j-l+I(1), i, k, mu, c] 
                
                @inbounds c_sc_buffB[I(1), j, i, k, c] -= (A_re * U_re - A_im * U_im)
            end
        end

        mult_complvec_by_complscal!(z, c_sc_buffB, result, i, k, i, k, i, k, c)
        
        return 
    end

    @inline function bwd_corrective_term!(
        root::CuDeviceArray{F, 4, 1}, inv_root::CuDeviceArray{F, 4, 1}, 
        z::CuDeviceArray{F, 6, 1}, U::CuDeviceArray{F, 6, 1}, result::CuDeviceArray{F, 6, 1}, 
        i::I, k::I, i_bwd::I, k_bwd::I, c::I,     
        nu_vac::CuDeviceArray{F, 4, 1}, M::CuDeviceArray{F, 7, 1}, mu::I, 
        c_sc_buffA::CuDeviceArray{F, 5, 1}, c_sc_buffB::CuDeviceArray{F, 5, 1}, sc_buffA::CuDeviceArray{F, 4, 1}
    )
        
        # Rapporto z^N(x-mu) / z^N(x) 
        # Firma: out(i,k), a(i_bwd, k_bwd), b(i,k)
        mult_sc_diffsites!(root, inv_root, sc_buffA, i_bwd, k_bwd, i, k, i, k, c)
        
        # Background e Matrice di Twist M^\dagger(x-mu) [Ordine g^0]
        idx_N = n_comps + I(1)
        @inbounds ratio_vac = nu_vac[I(1), i_bwd, k_bwd, idx_N] / nu_vac[I(1), i, k, idx_N]
        
        @inbounds M_re =  M[I(1), I(1), i_bwd, k_bwd, mu, idx_N, c]
        @inbounds M_im = -M[I(2), I(1), i_bwd, k_bwd, mu, idx_N, c] # Segno meno per l'Hermitiano coniugato

        for j::I = 1:n_ptords
            @inbounds val = sc_buffA[j, i, k, c] * ratio_vac
            @inbounds c_sc_buffA[I(1), j, i, k, c] = val * M_re
            @inbounds c_sc_buffA[I(2), j, i, k, c] = val * M_im
        end 

        # Moltiplicazione per U_mu^\dagger(x-mu)
        for j::I = 1:n_ptords
            @inbounds c_sc_buffB[I(1), j, i, k, c] = zero(F)
            @inbounds c_sc_buffB[I(2), j, i, k, c] = zero(F)
            
            for l::I = 1:j
                # Prefattore calcolato al passo precedente
                @inbounds A_re = c_sc_buffA[I(1), l, i, k, c]
                @inbounds A_im = c_sc_buffA[I(2), l, i, k, c]
                
                # Link di gauge valutato in x-mu 
                @inbounds U_re =  U[I(1), j-l+I(1), i_bwd, k_bwd, mu, c]
                @inbounds U_im = -U[I(2), j-l+I(1), i_bwd, k_bwd, mu, c] 
                
                @inbounds c_sc_buffB[I(1), j, i, k, c] -= (A_re * U_re - A_im * U_im) # --> siccome U è coniugato U_im ha il meno e quindi tornano i segni
            end
        end

        mult_complvec_by_complscal!(z, c_sc_buffB, result, i, k, i, k, i, k, c)
        
        return 
    end


    @inline function cmplx_scalar_prod!(z::CuDeviceArray{F, 6, 1}, conjMz::CuDeviceArray{F, 6, 1}, result::CuDeviceArray{F, 5, 1},
        i_a::I, k_a::I, i_b::I, k_b::I, i_out::I, k_out::I, c::I
    )
        for j::I = 1:n_ptords   
            @inbounds result[I(1), j, i_out, k_out, c] = zero(F)
            @inbounds result[I(2), j, i_out, k_out, c] = zero(F)  
            for n::I = 1:n_comps 
                for l::I = 1:j 
                    @inbounds z_bar_re = z[I(1), l, i_a, k_a, n, c]
                    @inbounds z_bar_im = -z[I(2), l, i_a, k_a, n, c]
                    @inbounds conjMz_re = conjMz[I(1), j-l+I(1), i_b, k_b, n, c]
                    @inbounds conjMz_im = conjMz[I(2), j-l+I(1), i_b, k_b, n, c]

                    @inbounds result[I(1), j, i_out, k_out, c] += (z_bar_re * conjMz_re - z_bar_im * conjMz_im)
                    @inbounds result[I(2), j, i_out, k_out, c] += (z_bar_re * conjMz_im + z_bar_im * conjMz_re) 
                end 

            end 
        end 
        return 
    end 

    @inline function cmplx_G1_func!(z::CuDeviceArray{F, 6, 1}, root::CuDeviceArray{F, 4, 1}, 
        M::CuDeviceArray{F, 7, 1}, nu_vac::CuDeviceArray{F, 4, 1}, G1::CuDeviceArray{F, 5, 1}, 
        i::I, k::I, i_fwd::I, k_fwd::I, mu::I, c::I, sc_buffA::CuDeviceArray{F, 4, 1}, vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )   
        
        mult_z_by_Mconj!(z, M, vec_buffer_comp, i, k, i, k, i, k, mu, c)
        cmplx_scalar_prod!(z, vec_buffer_comp, G1, i_fwd, k_fwd, i, k, i, k, c)
        
        idx_N = n_comps + I(1) 
        @inbounds prod_nu_fwd = nu_vac[I(1), i_fwd, k_fwd, idx_N] * nu_vac[I(1), i, k, idx_N] 
        @inbounds MconjRe = M[I(1), I(1), i, k, mu, idx_N, c]
        @inbounds MconjIm = -M[I(2), I(1), i, k, mu, idx_N, c]
        
        mult_sc_diffsites!(root, root, sc_buffA, i, k, i_fwd, k_fwd, i, k, c)
        for j::I = 1:n_ptords 
            @inbounds G1[I(1), j, i, k, c] += prod_nu_fwd * MconjRe * sc_buffA[j, i, k, c]
            @inbounds G1[I(2), j, i, k, c] += prod_nu_fwd * MconjIm * sc_buffA[j, i, k, c]
        end 

         
        return 
    end 

    function ener_kernel!(
        z::CuDeviceArray{F, 6, 1}, 
        U_mu::CuDeviceArray{F, 6, 1}, 
        ener::CuDeviceArray{F, 4, 1}, 
        root_zscaled2::CuDeviceArray{F, 4, 1}, 
        M1::CuDeviceArray{F, 7, 1}, 
        nu_vacuum::CuDeviceArray{F, 4, 1}, 
        c_sc_buffA::CuDeviceArray{F, 5, 1},
        sc_buffA::CuDeviceArray{F, 4, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )

        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            i_dw, k_dw = neighbour_down(i, k)  
            i_dx, k_dx = neighbour_right(i, k) 

            for j::I = 1:n_ptords
                @inbounds ener[j, i, k, c] = zero(F)
            end 
            
            for mu::I = 1:2
            # Assegnazione statica dei vicini forward (+mu) e backward (-mu)
                if mu == I(1)
                    # Direzione 1 (es. Temporale: up/down)
                    i_fwd, k_fwd = i_dw, k # Mi muovo avanti nella direzione temporale
                else
                    # Direzione 2 (es. Spaziale: right/left)
                    i_fwd, k_fwd = i, k_dx # Mi muovo avanti in quella spaziale 
                end 
                cmplx_G1_func!(z, root_zscaled2, M1, nu_vacuum, c_sc_buffA, i, k, i_fwd, k_fwd, mu, c, sc_buffA, vec_buffer_comp)
                
                @inbounds Uconj_mu_Re = U_mu[I(1), I(1), i, k, mu, c]
                @inbounds Uconj_mu_Im = -U_mu[I(2), I(1), i, k, mu, c]
                @inbounds G1_Re = c_sc_buffA[I(1), I(1), i, k, c]
                @inbounds G1_Im = c_sc_buffA[I(2), I(1), i, k, c]
                
                @inbounds ener[I(1), i, k, c] -= (Uconj_mu_Re * G1_Re - Uconj_mu_Im * G1_Im) - F(1) #In questo modo sottraggo 1 per mu = 1,2 

                for j::I=2:n_ptords 
                    for l::I = 1:j
                        @inbounds Uconj_mu_Re = U_mu[I(1), l, i, k, mu, c]
                        @inbounds Uconj_mu_Im = -U_mu[I(2), l, i, k, mu, c]
                        @inbounds G1_Re = c_sc_buffA[I(1), j-l+I(1), i, k, c]
                        @inbounds G1_Im = c_sc_buffA[I(2), j-l+I(1), i, k, c]

                        @inbounds ener[j, i, k, c] -= (Uconj_mu_Re * G1_Re - Uconj_mu_Im * G1_Im)
                    end 
                end 
            end 
        end
        return nothing
    end

    function grad_z_kernel!(
        z::CuDeviceArray{F, 6, 1}, 
        nu_vacuum::CuDeviceArray{F, 4, 1},
        U_mu::CuDeviceArray{F, 6, 1}, 
        gradient_z::CuDeviceArray{F, 6, 1}, 
        root_zscaled2::CuDeviceArray{F, 4, 1},
        invroot_zscaled2::CuDeviceArray{F, 4, 1},
        M1::CuDeviceArray{F, 7, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}, 
        c_sc_buffA::CuDeviceArray{F, 5, 1}, 
        c_sc_buffB::CuDeviceArray{F, 5, 1}, 
        sc_buffA::CuDeviceArray{F, 4, 1}
    )
        
        # Computes -grad_z ( visto che è quello che mi serve per Langevin)
        i, k, c = get_indexes()

        
        if (i <= Npoint && k <= Npoint && c <= n_copies) 

            i_up, k_up = neighbour_up(i, k) 
            i_dw, k_dw = neighbour_down(i, k) 
            i_dx, k_dx = neighbour_right(i, k) 
            i_sx, k_sx = neighbour_left(i, k) 

            for n::I = 1:n_comps 
                for j::I = 1:n_ptords 
                    @inbounds gradient_z[I(1), j, i, k, n, c] = zero(F)
                    @inbounds gradient_z[I(2), j, i, k, n, c] = zero(F)
                end 
            end 

            for mu::I = 1:2 # ---> mu = 1 è la direzione temporale 
    
            # Assegnazione statica dei vicini forward (+mu) e backward (-mu)
                if mu == I(1)
                    # Direzione 1 (es. Temporale: up/down)
                    i_fwd, k_fwd = i_dw, k
                    i_bwd, k_bwd = i_up, k
                else
                    # Direzione 2 (es. Spaziale: right/left)
                    i_fwd, k_fwd = i, k_dx
                    i_bwd, k_bwd = i, k_sx
                end 

                mult_z_by_Mconj!(z, M1, vec_buffer_comp, i_bwd, k_bwd, i_bwd, k_bwd, i, k, mu, c)
                mult_compvec_by_Uconj!(vec_buffer_comp, U_mu, gradient_z, i, k, i_bwd, k_bwd, i, k, mu, c)
                
                mult_z_by_M!(z, M1, vec_buffer_comp, i_fwd, k_fwd, i, k, i, k, mu, c)
                mult_compvec_by_U!(vec_buffer_comp, U_mu, gradient_z, i, k, i, k, i, k, mu, c)

            end 

            for mu::I = 1:2
            
                if mu == I(1)
                    # Direzione 1 (es. Temporale: up/down)
                    i_fwd, k_fwd = i_dw, k
                    i_bwd, k_bwd = i_up, k
                else
                    # Direzione 2 (es. Spaziale: right/left)
                    i_fwd, k_fwd = i, k_dx
                    i_bwd, k_bwd = i, k_sx
                end 
                
                
                fwd_corrective_term!(root_zscaled2, invroot_zscaled2, z, U_mu, gradient_z, i, k, i_fwd, k_fwd, c, nu_vacuum, M1, mu, c_sc_buffA, c_sc_buffB, sc_buffA)
                bwd_corrective_term!(root_zscaled2, invroot_zscaled2, z, U_mu, gradient_z, i, k, i_bwd, k_bwd, c, nu_vacuum, M1, mu, c_sc_buffA, c_sc_buffB, sc_buffA)
            
            
            end 


        end 
        return nothing
    end


    function grad_U_kernel!(
        grad_U_mu::CuDeviceArray{F, 5, 1}, 
        U_mu::CuDeviceArray{F, 6, 1}, 
        z::CuDeviceArray{F, 6, 1}, 
        M1::CuDeviceArray{F, 7, 1}, 
        root_zscaled2::CuDeviceArray{F, 4, 1},
        nu_vacuum::CuDeviceArray{F, 4, 1}, 
        c_sc_buffB::CuDeviceArray{F, 5, 1}, 
        sc_buffA::CuDeviceArray{F, 4, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )
        # Calcola -gradient_U
        i, k, c = get_indexes()

        if (i <= Npoint && k <= Npoint && c <= n_copies) 
            
            i_dw, k_dw = neighbour_down(i, k)  
            i_dx, k_dx = neighbour_right(i, k) 

            for mu::I = 1:2
            # Assegnazione statica dei vicini forward (+mu) e backward (-mu)
                if mu == I(1)
                    # Direzione 1 (es. Temporale: up/down)
                    i_fwd, k_fwd = i_dw, k
                else
                    # Direzione 2 (es. Spaziale: right/left)
                    i_fwd, k_fwd = i, k_dx
                end 
                cmplx_G1_func!(z, root_zscaled2, M1, nu_vacuum, c_sc_buffB, i, k, i_fwd, k_fwd, mu, c, sc_buffA, vec_buffer_comp)
                
                for j::I=1:n_ptords 
                    @inbounds grad_U_mu[j, i, k, mu, c] = zero(F)
                    for l::I = 1:j
                        @inbounds Uconj_mu_Re = U_mu[I(1), l, i, k, mu, c]
                        @inbounds Uconj_mu_Im = -U_mu[I(2), l, i, k, mu, c]
                        @inbounds G1_Re = c_sc_buffB[I(1), j-l+I(1), i, k, c]
                        @inbounds G1_Im = c_sc_buffB[I(2), j-l+I(1), i, k, c]

                        @inbounds grad_U_mu[j, i, k, mu, c] += Uconj_mu_Re * G1_Im + Uconj_mu_Im * G1_Re # la forza è solo la parte immaginaria 
                    end 
                end 
            end 
         end 

        return nothing 
    end 

    function zeromode_z_kernel!(z::CuDeviceArray{F, 6, 1}, zero_modo_z::CuDeviceArray{F, 4, 1})
        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            for n::I = 1:n_comps
                for j::I = 1:n_ptords
                    @inbounds z[I(1),j,i,k,n,c] -= zero_modo_z[I(1), j, n, c]
                    @inbounds z[I(2),j,i,k,n,c] -= zero_modo_z[I(2), j, n, c]
                end 
            end 
           
        end
        return nothing
    end

    function zeromode_X_kernel!(X::CuDeviceArray{F, 5, 1}, zero_modo_X::CuDeviceArray{F, 3, 1})
        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            for mu::I = 1:2 
                for j::I = 2:n_ptords
                    @inbounds X[j,i,k,mu,c] -= zero_modo_X[j, mu, c]
                end 
            end 
        end 
        return nothing 
    end 

    return (zscaled2_kernel!, grad_z_kernel!, grad_U_kernel!, ener_kernel!, init_M1_kernel!, roots_kernel!, ExpU_kernel!, zeromode_z_kernel!, zeromode_X_kernel!)
end
