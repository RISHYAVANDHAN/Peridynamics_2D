function [ PL ] = AssignVols( Corners ,  PL , L , Bvals , TOPflag )

	NoPs = size(PL,2);
    PD = PL(1).PD;

    tol = 1e-4;

    switch TOPflag

    case 'CUT'

        if ( PD==2 )

            A = Corners(1,:); % bottom left
            B = Corners(2,:); % bottom right
            C = Corners(3,:); % top right
            D = Corners(4,:); % top left

            R = Bvals(1);
            Xo = Bvals(2);
            Yo = Bvals(3);

    		for p = 1:NoPs

                X = PL(p).X;
                
                if ( (X(1)-A(1))<(-tol) || (X(1)-B(1))>(tol) || (X(2)-A(2))<(-tol) || (X(2)-D(2))>(tol) )

                	alpha = 0;

                elseif ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol || abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol )

                	if ( (abs(X(1)-A(1))<tol && abs(X(2)-A(2))<tol) || ...
                	     (abs(X(1)-B(1))<tol && abs(X(2)-B(2))<tol) || ...
                	     (abs(X(1)-C(1))<tol && abs(X(2)-C(2))<tol) || ...
                	     (abs(X(1)-D(1))<tol && abs(X(2)-D(2))<tol) )

                		alpha = 1/4;

                	else

                		alpha = 1/2;

                	end

                else

                    if ( (X(1) - (Xo+R))>tol || (X(1) - (Xo-R))<-tol || (X(2) - (Yo+R))>tol || (X(2) - (Yo-R))<-tol )

                    	alpha = 1;

                    elseif ( abs(X(1) - (Xo+R))<tol || abs(X(1) - (Xo-R))<tol || abs(X(2) - (Yo+R))<tol || abs(X(2) - (Yo-R))<tol )

                        if ( ( abs(X(1) - (Xo+R))<tol && abs(X(2) - (Yo-R))<tol ) || ...
                             ( abs(X(1) - (Xo+R))<tol && abs(X(2) - (Yo+R))<tol ) || ...
                             ( abs(X(1) - (Xo-R))<tol && abs(X(2) - (Yo-R))<tol ) || ...
                             ( abs(X(1) - (Xo-R))<tol && abs(X(2) - (Yo+R))<tol ) )

                            alpha = 3/4;

                        else

                            alpha = 1/2;

                        end

                    else

                        alpha = 0;

                    end

                end

    			V = alpha * L^2;

    			PL(p).Vol = V;

            end

        elseif ( PD==3 )

            A = Corners(1,:); % left south bottom corner
            B = Corners(2,:); % right south bottom corner
            C = Corners(3,:); % right north bottom corner
            D = Corners(4,:); % left north bottom corner
            E = Corners(5,:); % left south top corner
            F = Corners(6,:); % right south top corner
            G = Corners(7,:); % right north top corner
            H = Corners(8,:); % left north top corner

            R = Bvals(1);
            Xo = Bvals(2);
            Yo = Bvals(3);
            Zo = Bvals(4);

            for p = 1:NoPs

                X = PL(p).X;
                
                if ( (X(1)-A(1))<(-tol) || (X(1)-B(1))>(tol) || (X(2)-A(2))<(-tol) || (X(2)-D(2))>(tol) || (X(3)-A(3))<(-tol) || (X(3)-E(3))>(tol) )

                    alpha = 0;

                elseif ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol || abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol || abs(X(3)-A(3))<tol || abs(X(3)-E(3))<tol )

                    if ( (abs(X(1)-A(1))<tol && abs(X(2)-A(2))<tol) || ...
                         (abs(X(1)-B(1))<tol && abs(X(2)-B(2))<tol) || ...
                         (abs(X(1)-C(1))<tol && abs(X(2)-C(2))<tol) || ...
                         (abs(X(1)-D(1))<tol && abs(X(2)-D(2))<tol) )

                        if ( abs(X(3)-A(3))<tol || abs(X(3)-E(3))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    elseif ( (abs(X(1)-A(1))<tol && abs(X(3)-A(3))<tol) || ...
                             (abs(X(1)-B(1))<tol && abs(X(3)-B(3))<tol) || ...
                             (abs(X(1)-E(1))<tol && abs(X(3)-E(3))<tol) || ...
                             (abs(X(1)-F(1))<tol && abs(X(3)-F(3))<tol) )

                        if ( abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    elseif ( (abs(X(2)-A(2))<tol && abs(X(3)-A(3))<tol) || ...
                             (abs(X(2)-D(2))<tol && abs(X(3)-D(3))<tol) || ...
                             (abs(X(2)-E(2))<tol && abs(X(3)-E(3))<tol) || ...
                             (abs(X(2)-H(2))<tol && abs(X(3)-H(3))<tol) )

                        if ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    else

                        alpha = 1/2;

                    end

                else

                    if ( (X(1) - (Xo+R))>tol || (X(1) - (Xo-R))<-tol || (X(2) - (Yo+R))>tol || (X(2) - (Yo-R))<-tol || (X(3) - (Zo+R))>tol || (X(3) - (Zo-R))<-tol )

                        alpha = 1;

                    elseif ( abs(X(1) - (Xo+R))<tol || abs(X(1) - (Xo-R))<tol || abs(X(2) - (Yo+R))<tol || abs(X(2) - (Yo-R))<tol || abs(X(3) - (Zo+R))<tol || abs(X(3) - (Zo-R))<tol )

                        if ( (abs(X(1)-(Xo-R))<tol && abs(X(2)-(Yo-R))<tol) || ...
                             (abs(X(1)-(Xo+R))<tol && abs(X(2)-(Yo-R))<tol) || ...
                             (abs(X(1)-(Xo+R))<tol && abs(X(2)-(Yo+R))<tol) || ...
                             (abs(X(1)-(Xo-R))<tol && abs(X(2)-(Yo+R))<tol) )

                            if ( abs(X(3)-(Zo-R))<tol || abs(X(3)-(Zo+R))<tol  )

                                alpha = 7/8;

                            else

                                alpha = 3/4;

                            end

                        elseif ( (abs(X(1)-(Xo-R))<tol && abs(X(3)-(Zo-R))<tol) || ...
                                 (abs(X(1)-(Xo+R))<tol && abs(X(3)-(Zo-R))<tol) || ...
                                 (abs(X(1)-(Xo-R))<tol && abs(X(3)-(Zo+R))<tol) || ...
                                 (abs(X(1)-(Xo+R))<tol && abs(X(3)-(Zo+R))<tol) )

                            if ( abs(X(2)-(Yo-R))<tol || abs(X(2)-(Yo+R))<tol  )

                                alpha = 7/8;

                            else

                                alpha = 3/4;

                            end

                        elseif ( (abs(X(2)-(Yo-R))<tol && abs(X(3)-(Zo-R))<tol) || ...
                                 (abs(X(2)-(Yo+R))<tol && abs(X(3)-(Zo-R))<tol) || ...
                                 (abs(X(2)-(Yo-R))<tol && abs(X(3)-(Zo+R))<tol) || ...
                                 (abs(X(2)-(Yo+R))<tol && abs(X(3)-(Zo+R))<tol) )

                            if ( abs(X(1)-(Xo-R))<tol || abs(X(1)-(Xo+R))<tol  )

                                alpha = 7/8;

                            else

                                alpha = 3/4;

                            end

                        else

                            alpha = 1/2;

                        end

                    else

                        alpha = 0;

                    end

                end

                V = alpha * L^3;

                PL(p).Vol = V;

            end

    	end

    case 'FULL'

        if ( PD==2 )

            A = Corners(1,:); % bottom left
            B = Corners(2,:); % bottom right
            C = Corners(3,:); % top right
            D = Corners(4,:); % top left

            for p = 1:NoPs

                X = PL(p).X;
                
                if ( (X(1)-A(1))<(-tol) || (X(1)-B(1))>(tol) || (X(2)-A(2))<(-tol) || (X(2)-D(2))>(tol) )

                    alpha = 0;

                elseif ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol || abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol )

                    if ( (abs(X(1)-A(1))<tol && abs(X(2)-A(2))<tol) || ...
                         (abs(X(1)-B(1))<tol && abs(X(2)-B(2))<tol) || ...
                         (abs(X(1)-C(1))<tol && abs(X(2)-C(2))<tol) || ...
                         (abs(X(1)-D(1))<tol && abs(X(2)-D(2))<tol) )

                        alpha = 1/4;

                    else

                        alpha = 1/2;

                    end

                else

                    alpha = 1;

                end

                V = alpha * L^2;

                PL(p).Vol = V;

            end

        elseif ( PD==3 )

            A = Corners(1,:); % left south bottom corner
            B = Corners(2,:); % right south bottom corner
            C = Corners(3,:); % right north bottom corner
            D = Corners(4,:); % left north bottom corner
            E = Corners(5,:); % left south top corner
            F = Corners(6,:); % right south top corner
            G = Corners(7,:); % right north top corner
            H = Corners(8,:); % left north top corner

            for p = 1:NoPs

                X = PL(p).X;
                
                if ( (X(1)-A(1))<(-tol) || (X(1)-B(1))>(tol) || (X(2)-A(2))<(-tol) || (X(2)-D(2))>(tol) || (X(3)-A(3))<(-tol) || (X(3)-E(3))>(tol) )

                    alpha = 0;

                elseif ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol || abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol || abs(X(3)-A(3))<tol || abs(X(3)-E(3))<tol )

                    if ( (abs(X(1)-A(1))<tol && abs(X(2)-A(2))<tol) || ...
                         (abs(X(1)-B(1))<tol && abs(X(2)-B(2))<tol) || ...
                         (abs(X(1)-C(1))<tol && abs(X(2)-C(2))<tol) || ...
                         (abs(X(1)-D(1))<tol && abs(X(2)-D(2))<tol) )

                        if ( abs(X(3)-A(3))<tol || abs(X(3)-E(3))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    elseif ( (abs(X(1)-A(1))<tol && abs(X(3)-A(3))<tol) || ...
                             (abs(X(1)-B(1))<tol && abs(X(3)-B(3))<tol) || ...
                             (abs(X(1)-E(1))<tol && abs(X(3)-E(3))<tol) || ...
                             (abs(X(1)-F(1))<tol && abs(X(3)-F(3))<tol) )

                        if ( abs(X(2)-A(2))<tol || abs(X(2)-D(2))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    elseif ( (abs(X(2)-A(2))<tol && abs(X(3)-A(3))<tol) || ...
                             (abs(X(2)-D(2))<tol && abs(X(3)-D(3))<tol) || ...
                             (abs(X(2)-E(2))<tol && abs(X(3)-E(3))<tol) || ...
                             (abs(X(2)-H(2))<tol && abs(X(3)-H(3))<tol) )

                        if ( abs(X(1)-A(1))<tol || abs(X(1)-B(1))<tol  )

                            alpha = 1/8;

                        else

                            alpha = 1/4;

                        end

                    else

                        alpha = 1/2;

                    end

                else

                    alpha = 1;

                end

                V = alpha * L^3;

                PL(p).Vol = V;

            end

        end

    end

end
