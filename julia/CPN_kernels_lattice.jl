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
    #N_colors = n_comps + one(n_comps) total number of colors --> non serve più, basta dire che n_comps è il numero totale di componenti 

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

    function init_M1_kernel!(M1::CuDeviceArray{F, 7, 1}) 
        
        i, k, c = get_indexes() 
        
        if (i <= Npoint && k <= Npoint && c <= n_copies)

            
            for mu::I = 1:2 # 1 è la direzione temporale, 2 è la direzione spaziale. 
                for n::I = 1:n_comps
                    for j::I = 1:n_ptords
                        @inbounds M1[I(1), j, i, k, mu, n, c] = zero(F)
                        @inbounds M1[I(2), j, i, k, mu, n, c] = zero(F)
                    end
                end
            end
            
            for mu::I = 1:2
                for n::I = 1:n_comps
                    # Calcolo della fase: in Julia n va da 1 a N_colors, nel C era k da 0 a N-1
                    phase = F(2.0 * pi * (n - 1) / n_comps)
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

    @inline function cmplx_G1_func!(z::CuDeviceArray{F, 6, 1}, M::CuDeviceArray{F, 7, 1}, G1::CuDeviceArray{F, 5, 1}, 
        i::I, k::I, i_fwd::I, k_fwd::I, mu::I, c::I, vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )   
        
        mult_z_by_Mconj!(z, M, vec_buffer_comp, i, k, i, k, i, k, mu, c)
        cmplx_scalar_prod!(z, vec_buffer_comp, G1, i_fwd, k_fwd, i, k, i, k, c)
         
        return 
    end

    function ener_kernel!(
        z::CuDeviceArray{F, 6, 1}, 
        U_mu::CuDeviceArray{F, 6, 1}, 
        ener::CuDeviceArray{F, 4, 1}, 
        M1::CuDeviceArray{F, 7, 1}, 
        c_sc_buffA::CuDeviceArray{F, 5, 1},
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
                cmplx_G1_func!(z, M1, c_sc_buffA, i, k, i_fwd, k_fwd, mu, c, vec_buffer_comp)
                
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

    # Nuova funzione per il gradiente--> Ho tenuto semplicemente il primo pezzo di quella vecchia 
    # Devo solo stare attento perché ora n_comps = N e non più N_comps = N-1 !!! 
    @inline function parallel_component!(
        cmplx_vec::CuDeviceArray{F, 6, 1},
        z::CuDeviceArray{F, 6, 1},
        c_sc_buffA::CuDeviceArray{F, 5, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}, 
        i::I, k::I, c::I
    )

        for j::I = 1:n_ptords 
            @inbounds c_sc_buffA[I(1), j, i, k, c] = zero(F)
            @inbounds c_sc_buffA[I(2), j, i, k, c] = zero(F)

            for n::I = 1:n_comps
                @inbounds nu_real = z[I(1), I(1), i, k, n, c]
                @inbounds nu_imm = z[I(2), I(1), i, k, n, c]
                @inbounds vec_real = cmplx_vec[I(1), j, i, k, n, c]
                @inbounds vec_imm = cmplx_vec[I(2), j, i, k, n, c]

                @inbounds c_sc_buffA[I(1),j,i,k,c] += nu_real * vec_real + nu_imm * vec_imm
                @inbounds c_sc_buffA[I(2),j,i,k,c] += nu_real * vec_imm - nu_imm * vec_real 
            end 
        end 

        for j::I = 1:n_ptords 
            for n::I = 1:n_comps
                @inbounds nu_real = z[I(1), I(1), i, k, n, c]
                @inbounds nu_imm = z[I(2), I(1), i, k, n, c]
                @inbounds scal_re = c_sc_buffA[I(1), j, i, k, c]
                @inbounds scal_im = c_sc_buffA[I(2), j, i, k, c]

                @inbounds vec_buffer_comp[I(1), j, i, k, n, c] = nu_real * scal_re - nu_imm * scal_im
                @inbounds vec_buffer_comp[I(2), j, i, k, n, c] = nu_imm * scal_re + nu_real * scal_im

                @inbounds cmplx_vec[I(1), j, i, k, n, c] -= vec_buffer_comp[I(1), j, i, k, n, c]
                @inbounds cmplx_vec[I(2), j, i, k, n, c] -= vec_buffer_comp[I(2), j, i, k, n, c]
            end 
        end 
        return 
    end

    @inline function gradient_projection!(
        grad_vec::CuDeviceArray{F, 6, 1},
        z::CuDeviceArray{F, 6, 1},
        c_sc_buffA::CuDeviceArray{F, 5, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}, 
        i::I, k::I, c::I
    )

        for j::I = 1:n_ptords 
            for n::I = 1:n_comps 
                @inbounds vec_buffer_comp[I(1),j,i,k,n,c] = zero(F)
                @inbounds vec_buffer_comp[I(2),j,i,k,n,c] = zero(F)
            end 
        end 
        cmplx_scalar_prod!(z, grad_vec, c_sc_buffA, i, k, i, k, i, k, c)
        mult_complvec_by_complscal!(z, c_sc_buffA, vec_buffer_comp, i, k, i, k, i, k, c)

        for j::I = 1:n_ptords 
            for n::I = 1:n_comps 
                @inbounds grad_vec[I(1),j,i,k,n,c] -= vec_buffer_comp[I(1),j,i,k,n,c]
                @inbounds grad_vec[I(2),j,i,k,n,c] -= vec_buffer_comp[I(2),j,i,k,n,c]
            end 
        end 
        
        return 
    end


    function grad_z_kernel!(
        z::CuDeviceArray{F, 6, 1}, 
        U_mu::CuDeviceArray{F, 6, 1}, 
        gradient_z::CuDeviceArray{F, 6, 1}, 
        M1::CuDeviceArray{F, 7, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}, 
        c_sc_buffA::CuDeviceArray{F, 5, 1}
    )
        
        # Computes the parallel component of -grad_z ( visto che è quello che mi serve per Langevin)
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


            gradient_projection!(gradient_z, z, c_sc_buffA, vec_buffer_comp, i, k, c)


        end 
        return nothing
    end


    function grad_U_kernel!(
        grad_U_mu::CuDeviceArray{F, 5, 1}, 
        U_mu::CuDeviceArray{F, 6, 1}, 
        z::CuDeviceArray{F, 6, 1}, 
        M1::CuDeviceArray{F, 7, 1}, 
        c_sc_buffB::CuDeviceArray{F, 5, 1}, 
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
                cmplx_G1_func!(z, M1, c_sc_buffB, i, k, i_fwd, k_fwd, mu, c, vec_buffer_comp)
                
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

    function correct_z_kernel!(
        z::CuDeviceArray{F, 6, 1}, 
        sc_buffA::CuDeviceArray{F, 4, 1},
        vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )

        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            
            @inbounds sc_buffA[I(1),i,k,c] = zero(F)
            for j::I = 2:n_ptords 
                sc_buffA[j,i,k,c] = zero(F)
                for n::I = 1:n_comps 
                    @inbounds nu_real = z[I(1),I(1),i,k,n,c]
                    @inbounds nu_imm = z[I(2),I(1),i,k,n,c]
                    @inbounds z_real = z[I(1),j,i,k,n,c]
                    @inbounds z_imm = z[I(2),j,i,k,n,c]

                    @inbounds sc_buffA[j,i,k,c] -= nu_real * z_real + nu_imm * z_imm
                end 

                for l::I = 2:(j-1) 
                    for n::I = 1:n_comps 
                        @inbounds z1_real = z[I(1),l,i,k,n,c]
                        @inbounds z1_imm = z[I(2),l,i,k,n,c]
                        @inbounds z2_real = z[I(1),j-l+I(1),i,k,n,c]
                        @inbounds z2_imm = z[I(2),j-l+I(1),i,k,n,c]

                        @inbounds sc_buffA[j,i,k,c] -= F(0.5) * (z1_real * z2_real + z1_imm * z2_imm)
                    end 
                end 
            
            end 
            
            for j::I = 1:n_ptords 
                for n::I = 1:n_comps 
                    @inbounds nu_real = z[I(1),I(1),i,k,n,c]
                    @inbounds nu_imm = z[I(2),I(1),i,k,n,c]
                    @inbounds alpha = sc_buffA[j,i,k,c]

                    @inbounds z[I(1),j,i,k,n,c] += alpha * nu_real 
                    @inbounds z[I(2),j,i,k,n,c] += alpha * nu_imm
                end 
            end 

        end 
        
        return nothing 
    end 

    function project_noise_kernel!(
        noise::CuDeviceArray{F, 6, 1},
        z::CuDeviceArray{F, 6, 1},
        c_sc_buffA::CuDeviceArray{F, 5, 1}, 
        vec_buffer_comp::CuDeviceArray{F, 6, 1}
    )
        i, k, c = get_indexes()
        
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            # Applica la proiezione radiale al vettore di rumore corrente
            parallel_component!(noise, z, c_sc_buffA, vec_buffer_comp, i, k, c)
        end 
        
        return nothing
    end

    # Calcola la trasformazione di gauge nel gauge di backgrounds 
    # Ebar * E = 1 fissa la parte reale ( fai il conto con lo sviluppo perturbativo)
    # Im(v_bar_N * z_N') = 0 !! ( dove z_N' è il trasformato di gauge per l'N-esima componente del vettore z sviluppato e con 
    # |z(x)|^2 = 1)
    function compute_gauge_transform_kernel!(
        z::CuDeviceArray{F, 6, 1},
        E_field::CuDeviceArray{F, 5, 1} # Useremo c_sc_buffA per questo
    )
        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            # All'ordine 0, la trasformazione è l'identità (non tocchiamo il vuoto)
            @inbounds E_field[I(1), I(1), i, k, c] = one(F)
            @inbounds E_field[I(2), I(1), i, k, c] = zero(F)
            
            # Leggiamo il background classico (ordine 1 nell'array)
            @inbounds nu_N_re = z[I(1), I(1), i, k, n_comps, c]
            @inbounds nu_N_im = z[I(2), I(1), i, k, n_comps, c]
            nu_N_sq = nu_N_re^2 + nu_N_im^2
            
            # Costruiamo E ricorsivamente ordine per ordine
            for j::I = 2:n_ptords
                # 1. Condizione di unitarità (fissa la parte reale di E)
                S_val = zero(F)
                for l::I = 2:(j-1)
                    @inbounds E1_re = E_field[I(1), l, i, k, c]
                    @inbounds E1_im = E_field[I(2), l, i, k, c]
                    @inbounds E2_re = E_field[I(1), j-l+I(1), i, k, c]
                    @inbounds E2_im = -E_field[I(2), j-l+I(1), i, k, c] # complesso coniugato
                    S_val += E1_re * E2_re - E1_im * E2_im
                end
                @inbounds E_field[I(1), j, i, k, c] = -F(0.5) * S_val # Fai i conti con gli sviluppi !! 
                
                # 2. Condizione di Gauge Cooling (fissa la parte immaginaria di E)
                K_re = zero(F)
                K_im = zero(F)
                for l::I = 1:(j-1)
                    @inbounds El_re = E_field[I(1), l, i, k, c]
                    @inbounds El_im = E_field[I(2), l, i, k, c]
                    @inbounds z_re = z[I(1), j-l+I(1), i, k, n_comps, c]
                    @inbounds z_im = z[I(2), j-l+I(1), i, k, n_comps, c]
                    
                    K_re += El_re * z_re - El_im * z_im # K è il pezzo del prodotto che non dipende dal E^(j)
                    K_im += El_im * z_re + El_re * z_im
                end
                
                im_part = nu_N_re * K_im - nu_N_im * K_re
                @inbounds E_field[I(2), j, i, k, c] = - im_part / nu_N_sq
            end
        end
        return nothing
    end

    function apply_gauge_transform_kernel!(
        z::CuDeviceArray{F, 6, 1},
        U_mu::CuDeviceArray{F, 6, 1},
        E_field::CuDeviceArray{F, 5, 1},
        vec_buffer_comp::CuDeviceArray{F, 6, 1} 
    )
        i, k, c = get_indexes()
        if (i <= Npoint && k <= Npoint && c <= n_copies)
            
            i_dx, k_dx = neighbour_right(i, k) 
            i_dw, k_dw = neighbour_down(i, k) 
            
            # 1. Ruota z(x) -> E(x) * z(x)
            for n::I = 1:n_comps
                for j::I = 1:n_ptords
                    res_re = zero(F)
                    res_im = zero(F)
                    for l::I = 1:j
                        @inbounds E_re = E_field[I(1), l, i, k, c]
                        @inbounds E_im = E_field[I(2), l, i, k, c]
                        @inbounds z_re = z[I(1), j-l+I(1), i, k, n, c]
                        @inbounds z_im = z[I(2), j-l+I(1), i, k, n, c]
                        res_re += E_re * z_re - E_im * z_im
                        res_im += E_im * z_re + E_re * z_im
                    end
                    @inbounds vec_buffer_comp[I(1), j, i, k, n, c] = res_re
                    @inbounds vec_buffer_comp[I(2), j, i, k, n, c] = res_im
                end
            end
            
            for n::I = 1:n_comps
                for j::I = 1:n_ptords
                    @inbounds z[I(1), j, i, k, n, c] = vec_buffer_comp[I(1), j, i, k, n, c]
                    @inbounds z[I(2), j, i, k, n, c] = vec_buffer_comp[I(2), j, i, k, n, c]
                end
            end
            
            # 2. Ruota i link U_mu(x) -> E(x) * U_mu(x) * conj(E(x+mu))
            for mu::I = 1:2
                if mu == I(1)
                    i_fwd, k_fwd = i_dw, k
                else
                    i_fwd, k_fwd = i, k_dx
                end
                
                # Passo A: T = E(x) * U_mu
                for j::I = 1:n_ptords
                    T_re = zero(F)
                    T_im = zero(F)
                    for l::I = 1:j
                        @inbounds E_re = E_field[I(1), l, i, k, c]
                        @inbounds E_im = E_field[I(2), l, i, k, c]
                        @inbounds U_re = U_mu[I(1), j-l+I(1), i, k, mu, c]
                        @inbounds U_im = U_mu[I(2), j-l+I(1), i, k, mu, c]
                        T_re += E_re * U_re - E_im * U_im
                        T_im += E_im * U_re + E_re * U_im
                    end
                    @inbounds vec_buffer_comp[I(1), j, i, k, mu, c] = T_re
                    @inbounds vec_buffer_comp[I(2), j, i, k, mu, c] = T_im
                end
                
                # Passo B: U_new = T * conj(E(x+mu))
                for j::I = 1:n_ptords
                    res_re = zero(F)
                    res_im = zero(F)
                    for l::I = 1:j
                        @inbounds T_re = vec_buffer_comp[I(1), l, i, k, mu, c]
                        @inbounds T_im = vec_buffer_comp[I(2), l, i, k, mu, c]
                        @inbounds Ef_re = E_field[I(1), j-l+I(1), i_fwd, k_fwd, c]
                        @inbounds Ef_im = -E_field[I(2), j-l+I(1), i_fwd, k_fwd, c] # coniugato!
                        res_re += T_re * Ef_re - T_im * Ef_im
                        res_im += T_im * Ef_re + T_re * Ef_im
                    end
                    # Salviamo in una zona sicura del buffer temporaneo
                    @inbounds vec_buffer_comp[I(1), j, i, k, mu+2, c] = res_re
                    @inbounds vec_buffer_comp[I(2), j, i, k, mu+2, c] = res_im
                end
                
                # Scriviamo definitivamente in U_mu
                for j::I = 1:n_ptords
                    @inbounds U_mu[I(1), j, i, k, mu, c] = vec_buffer_comp[I(1), j, i, k, mu+2, c]
                    @inbounds U_mu[I(2), j, i, k, mu, c] = vec_buffer_comp[I(2), j, i, k, mu+2, c]
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

    return (grad_z_kernel!, grad_U_kernel!, ener_kernel!, init_M1_kernel!, correct_z_kernel!, project_noise_kernel!, ExpU_kernel!, zeromode_z_kernel!)
end
