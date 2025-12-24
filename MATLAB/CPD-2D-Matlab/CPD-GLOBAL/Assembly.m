function out = Assembly( PL , Delta , DOFs , LF , flag)

    NoPs = size(PL,2);
    PD = PL(1).PD;

    switch flag

    case 'point-volume'

        out = zeros(NoPs,3);

        for p = 1 : NoPs

            out(p,:) = PL(p).volumes;

        end

    case 'volume'

        Vol = 0;

        for p = 1 : NoPs

            Vol_P = PL(p).Vol;

            Vol = Vol + Vol_P;

        end

        out = Vol;

    case 'energy'

        Psi = 0;

        for p = 1 : NoPs

            Psi_P = PL(p).energy;

            Vol_P = PL(p).Vol;

            Psi = Psi + Vol_P * Psi_P;

        end

        out = Psi;

    case 'residual'

        R = zeros(DOFs,1);

        for i = 1 : NoPs

            R_P = PL(i).residual;

            BCflg = PL(i).BCflg;
            DOF = PL(i).DOF;

            for ii = 1:PD

                if ( BCflg(ii)==1 )

                    R(DOF(ii)) = R(DOF(ii)) + R_P(ii);

                end

            end

        end

        out = R;

    case 'stiffness'

        % K = zeros(DOFs,DOFs); % will be sparse!!!

        sprC = 0; % sparse counter

        for p = 1:NoPs

            K_P = PL(p).stiffness;

            BCflg_p = PL(p).BCflg;
            DOF_p = PL(p).DOF;

            for pp = 1:PD

                if ( BCflg_p(pp)==1 )

                    nbrL = [PL(p).neighbors , p]; % append the point to its neighbors!
                        
                    for q = 1:length(nbrL)

                        K_PQtmp = K_P(:,q);

                        if ( PD==2 )
                            K_PQ = [ K_PQtmp(1) K_PQtmp(3) ; K_PQtmp(2) K_PQtmp(4) ];
                        elseif ( PD==3 )
                            K_PQ = [ K_PQtmp(1) K_PQtmp(4) K_PQtmp(7) ; K_PQtmp(2) K_PQtmp(5) K_PQtmp(8) ; K_PQtmp(3) K_PQtmp(6) K_PQtmp(9) ];
                        end

                        BCflg_q = PL(nbrL(q)).BCflg;
                        DOF_q = PL(nbrL(q)).DOF;
                        
                        for qq = 1:PD

                            if (BCflg_q(qq)==1)
                                sprC = sprC + 1;
                                ii(sprC) = DOF_p(pp);
                                jj(sprC) = DOF_q(qq);
                                vv(sprC) = K_PQ(pp,qq);
                            end

                        end % end of loop over qq

                    end % end of loop over q

                end
                    
            end % end of loop over pp
                
        end % end of loop over p

        K = sparse(ii,jj,vv);

        out = K;

    end

end