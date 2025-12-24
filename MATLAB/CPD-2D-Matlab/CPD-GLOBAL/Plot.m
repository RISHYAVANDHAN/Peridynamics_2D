function Plot(NL,flag,EL)

    if ( nargin==2 )
        EL = 0;
    end

    switch flag

    case 'mesh'

        PD = size(NL,2);

        if ( nargin==3 )

            if ( PD == 2 )

                NoEs = size(EL,1);
                NPE = size(EL,2);

                hold on;

                for e = 1 : NoEs

                    A = (NL(EL(e,1),:))';
                    B = (NL(EL(e,2),:))';
                    C = (NL(EL(e,3),:))';
                    D = (NL(EL(e,4),:))';

                    AB = [A,B];
                    BC = [B,C];
                    CD = [C,D];
                    DA = [D,A];

                    plot( AB(1,:) , AB(2,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot( BC(1,:) , BC(2,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot( CD(1,:) , CD(2,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot( DA(1,:) , DA(2,:), 'LineWidth' , 4 , 'Color' , 'k' );

                    x=(A(1)+B(1)+C(1)+D(1))/4;
                    y=(A(2)+B(2)+C(2)+D(2))/4;

                    text(x,y,num2str(e),'Color','b','FontSize',10,'HorizontalAlignment','center')
                end

                axis equal

                hold off

            elseif ( PD == 3 )

                NoEs = size(EL,1);
                NPE = size(EL,2);

                hold on;

                for e = 1 : NoEs

                    A = (NL(EL(e,1),:))';
                    B = (NL(EL(e,2),:))';
                    C = (NL(EL(e,3),:))';
                    D = (NL(EL(e,4),:))';
                    E = (NL(EL(e,5),:))';
                    F = (NL(EL(e,6),:))';
                    G = (NL(EL(e,7),:))';
                    H = (NL(EL(e,8),:))';

                    AB = [A,B];
                    BC = [B,C];
                    CD = [C,D];
                    DA = [D,A];
                    EF = [E,F];
                    FG = [F,G];
                    GH = [G,H];
                    HE = [H,E];
                    AE = [A,E];
                    BF = [B,F];
                    CG = [C,G];
                    DH = [D,H];

                    plot3( AB(1,:) , AB(2,:), AB(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( BC(1,:) , BC(2,:), BC(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( CD(1,:) , CD(2,:), CD(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( DA(1,:) , DA(2,:), DA(3,:), 'LineWidth' , 4 , 'Color' , 'k' );

                    plot3( EF(1,:) , EF(2,:), EF(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( FG(1,:) , FG(2,:), FG(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( GH(1,:) , GH(2,:), GH(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( HE(1,:) , HE(2,:), HE(3,:), 'LineWidth' , 4 , 'Color' , 'k' );

                    plot3( AE(1,:) , AE(2,:), AE(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( BF(1,:) , BF(2,:), BF(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( CG(1,:) , CG(2,:), CG(3,:), 'LineWidth' , 4 , 'Color' , 'k' );
                    plot3( DH(1,:) , DH(2,:), DH(3,:), 'LineWidth' , 4 , 'Color' , 'k' );

                    x=(A(1)+B(1)+C(1)+D(1)+E(1)+F(1)+G(1)+H(1))/8;
                    y=(A(2)+B(2)+C(2)+D(2)+E(2)+F(2)+G(2)+H(2))/8;
                    z=(A(3)+B(3)+C(3)+D(3)+E(3)+F(3)+G(3)+H(3))/8;

                    text(x,y,z,num2str(e),'Color','b','FontSize',10,'HorizontalAlignment','center')

                end

                axis equal

                hold off

            end

        end

    case 'nodes'

        NoNs = size(NL,1);
        PD = size(NL,2);

        if ( PD == 2 )

            hold on;

            for i = 1 : NoNs

                plot(NL(i,1),NL(i,2),'o','MarkerSize',18,'MarkerEdgeColor','r','MarkerFaceColor','r');
                text(NL(i,1),NL(i,2),num2str(i),'Color','black','FontSize',12,'HorizontalAlignment','center');

            end

            axis equal
            
            hold off

        elseif ( PD == 3 )

            hold on;

            for i = 1 : NoNs

                plot3(NL(i,1),NL(i,2),NL(i,3),'o','MarkerSize',12,'MarkerEdgeColor','r','MarkerFaceColor','r');
                text(NL(i,1),NL(i,2),NL(i,3),num2str(i),'Color','black','FontSize',12,'HorizontalAlignment','center');

            end

            axis equal
            
            hold off

        end

    case 'points'

        NoPs = size(NL,2);
        PD = NL(1).PD;

        if ( PD == 2 )

            hold on;

            for i = 1 : NoPs

                COOR = NL(i).X;
                coor = NL(i).x;

                % plot(COOR(1),COOR(2),'o','MarkerSize',2,'MarkerEdgeColor','r','MarkerFaceColor','r');
                plot(coor(1),coor(2),'o','MarkerSize',5,'MarkerEdgeColor','r','MarkerFaceColor','r');

            end

            axis equal
            
            hold off

        elseif ( PD == 3 )

            hold on;

            for i = 1 : NoPs

                COOR = NL(i).X;
                coor = NL(i).x;

                % plot3(COOR(1),COOR(2),COOR(3),'o','MarkerSize',2,'MarkerEdgeColor','r','MarkerFaceColor','r');
                plot3(coor(1),coor(2),coor(3),'o','MarkerSize',2,'MarkerEdgeColor','r','MarkerFaceColor','r');

            end

            axis equal
            
            hold off

        end

    end

end