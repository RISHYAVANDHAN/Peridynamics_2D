function [ PL ] = Update( PL , arg , flag )

    % UPDATE function

    NoPs = size(PL,2);
    PD = PL(1).PD;

    switch flag

    case 'prescribed'

        LF = arg;

        for i = 1:NoPs

            BCflg = PL(i).BCflg;
            BCval = PL(i).BCval;
            x = PL(i).x;
            X = PL(i).X;
            
            for p = 1:PD
                if ( BCflg(p) == 0 )
                    x(p) = X(p) + LF * BCval(p);
                end
            end

            PL(i).x = x;

        end

    case 'displacement'

        dx = arg;

        for i = 1:NoPs

            BCflg = PL(i).BCflg;
            DOF = PL(i).DOF;
            x = PL(i).x;

            for p = 1:PD
                if ( BCflg(p) == 1 )
                    x(p) = x(p) + dx(DOF(p));
                end
            end

            PL(i).x = x;

        end

    end % end of switch flag 

    for i = 1:NoPs

        nbrL = PL(i).neighbors;

        for j = 1:length(nbrL)

            neighborsx(:,j) = PL(nbrL(j)).x;

        end

        PL(i).neighborsx = neighborsx;

    end

end
