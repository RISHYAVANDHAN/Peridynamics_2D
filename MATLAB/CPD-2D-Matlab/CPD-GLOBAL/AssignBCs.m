function [ PL , DOFs ] = AssignBCs( Corners , PL , FF , BCflag , PatchFlag )

    NoPs = size(PL,2);
    PD = PL(1).PD;

    if ( PD==2 )

        A = Corners(1,:); % bottom left
        B = Corners(2,:); % bottom right
        C = Corners(3,:); % top right
        D = Corners(4,:); % top left

        switch BCflag

        case 'STD'

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            switch PatchFlag

            case 'fullpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) || (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            case 'horzpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            case 'vertpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            end

        case 'XTD'

            CNTR = [ 0 , 0 ];

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            for i = 1:NoPs

                X = PL(i).X;
                
                BCflg = zeros(PD,1);
                BCflg(1,1) = 0;
                BCflg(2,1) = 1;

                BCval = FF * X - X; 

                PL(i).BCflg = BCflg;
                PL(i).BCval = BCval;

                if ( abs(X(2)-0)<tol ) 

                    BCflg = zeros(PD,1);

                    BCval = FF * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end
            
            end

            % for i = 1:NoPs

            %     X = PL(i).X;
                
            %     if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) )

            %         BCflg = zeros(PD,1);
            %         BCflg(1,1) = 0;
            %         BCflg(2,1) = 1;

            %         BCval = FF * X - X; 

            %         PL(i).BCflg = BCflg;
            %         PL(i).BCval = BCval;

            %     end

            %     if ( abs(X(1)-CNTR(1))<tol && abs(X(2)-CNTR(2))<tol ) % at point A

            %         BCflg = zeros(PD,1);

            %         BCval = FF * X - X; 

            %         PL(i).BCflg = BCflg;
            %         PL(i).BCval = BCval;

            %     end
            
            % end

        case 'XTM'

            CNTR = [ 0 , 0 ];

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            % for i = 1:NoPs

            %     X = PL(i).X;
                
            %     BCflg = zeros(PD,1);
            %     BCflg(1,1) = 0;
            %     BCflg(2,1) = 1;

            %     BCval = FF * X - X; 

            %     PL(i).BCflg = BCflg;
            %     PL(i).BCval = BCval;

            %     if ( abs(X(2)-0)<tol ) 

            %         BCflg = zeros(PD,1);

            %         BCval = FF * X - X; 

            %         PL(i).BCflg = BCflg;
            %         PL(i).BCval = BCval;

            %     end
            
            % end

            FFM = FF;
            FFM(2,2) = 1-1/3*(FF(1,1)-1);

            for i = 1:NoPs

                X = PL(i).X;
                
                if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) )

                    BCflg = zeros(PD,1);

                    BCval = FFM * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end

            end

        case 'DBC' % PatchFlag = 'fullpatch' ... no need to check

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            for i = 1:NoPs

                X = PL(i).X;
                
                if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) || (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) ) % 'fullpatch'

                    BCflg = zeros(PD,1);

                    BCval = FF * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end
                
            end

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

        switch BCflag

        case 'STD'

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            switch PatchFlag

            case 'fullpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) || (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) || (X(3)-A(3))<tol || (X(3)-E(3))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            case 'horzpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            case 'vertpatch'

                for i = 1:NoPs

                    X = PL(i).X;
                    
                    if ( (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) )

                        BCflg = zeros(PD,1);

                        BCval = FF * X - X; 

                        PL(i).BCflg = BCflg;
                        PL(i).BCval = BCval;

                    end
                
                end

            end

        case 'XTD'

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            for i = 1:NoPs

                X = PL(i).X;
                
                if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) )

                    BCflg = zeros(PD,1);
                    BCflg(1,1) = 0;
                    BCflg(2,1) = 1;
                    BCflg(3,1) = 1;

                    BCval = FF * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end

                if ( abs(X(1)-A(1))<tol && abs(X(2)-A(2))<tol && abs(X(3)-A(3))<tol ) % at point A

                    BCflg = zeros(PD,1);

                    BCval = FF * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end
            
            end

        case 'DBC' % PatchFlag = 'fullpatch' ... no need to check

            PL = FreeAllPoints( PL );

            tol = 1e-3;

            for i = 1:NoPs

                X = PL(i).X;

                if ( (X(1)-A(1))<tol || (X(1)-B(1))>(-tol) || (X(2)-A(2))<tol || (X(2)-D(2))>(-tol) || (X(3)-A(3))<tol || (X(3)-E(3))>(-tol) )
                
                    BCflg = zeros(PD,1);

                    BCval = FF * X - X; 

                    PL(i).BCflg = BCflg;
                    PL(i).BCval = BCval;

                end
                
            end

        end

    end

    [ PL , DOFs ] = AssignGlobalDOF( PL );

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       FREE ALL POINTS       %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ PL ] = FreeAllPoints( PL )

    PD = PL(1).PD;

    % free all points with no force ... prescribe homogeneous Neumann BC on all points

    NoPs = size(PL,2);

    BCflg = zeros(PD,1);
    BCval = zeros(PD,1);

    for i = 1:PD
        BCflg(i) = 1;
        BCval(i) = 0;
    end

    for i = 1:NoPs      
        PL(i).BCflg = BCflg;
        PL(i).BCval = BCval;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%       ASSING DEGREES OF FREEDOM       %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ PL , DOFs ] = AssignGlobalDOF( PL )

    PD = PL(1).PD;

    NoPs = size(PL,2);

    DOFs = 0;

    for i = 1:NoPs
        
        BCflg = PL(i).BCflg;
        
        DOF = zeros(PD,1);

        for p = 1:PD

            if ( BCflg(p) == 1 )
                DOFs = DOFs+1;
                DOF(p) = DOFs;
            end

        end

        PL(i).DOF = DOF;
        
    end

end
