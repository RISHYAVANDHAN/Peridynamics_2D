function [ NL ] = Patch( Corners , L , Delta , PatchFlag ) 

	PD = size(Corners(1,:),2);

	v = round(Delta/L)*L;

	count = 0;

	if ( PD == 2 )

		Corners_mod(1,:) = Corners(1,:) + v*[ -1 , -1 ]; % left south corner
		Corners_mod(2,:) = Corners(2,:) + v*[ 1 , -1 ]; % right south corner
		Corners_mod(3,:) = Corners(3,:) + v*[ 1 , 1 ]; % right north corner
		Corners_mod(4,:) = Corners(4,:) + v*[ -1 , 1 ]; % left north corner

		[ NLtmp ] = Mesh( Corners_mod , L );

        NoNs = size(NLtmp,1);

        for i = 1 : NoNs

        	node = [ NLtmp(i,1) , NLtmp(i,2) ];

            if PatchNode( node , Corners , PatchFlag )

            	count = count + 1;

            	NL(count,:) = NLtmp(i,:);

            end

        end

	elseif ( PD == 3 )

		Corners_mod(1,:) = Corners(1,:) + v*[ -1 , -1 , -1 ]; % left south bottom corner
		Corners_mod(2,:) = Corners(2,:) + v*[ 1 , -1 , -1 ]; % right south bottom corner
		Corners_mod(3,:) = Corners(3,:) + v*[ 1 , 1 , -1 ]; % right north bottom corner
		Corners_mod(4,:) = Corners(4,:) + v*[ -1 , 1 , -1 ]; % left north bottom corner
		Corners_mod(5,:) = Corners(5,:) + v*[ -1 , -1 , 1 ]; % left south top corner
		Corners_mod(6,:) = Corners(6,:) + v*[ 1 , -1 , 1 ]; % right south top corner
		Corners_mod(7,:) = Corners(7,:) + v*[ 1 , 1 , 1 ]; % right north top corner
		Corners_mod(8,:) = Corners(8,:) + v*[ -1 , 1 , 1 ]; % left north top corner

		[ NLtmp ] = Mesh( Corners_mod , L );

        NoNs = size(NLtmp,1);

        for i = 1 : NoNs

        	node = [ NLtmp(i,1) , NLtmp(i,2) , NLtmp(i,3) ];

            if PatchNode( node , Corners , PatchFlag )

            	count = count + 1;

            	NL(count,:) = NLtmp(i,:);

            end

        end

	end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = PatchNode( node , Corners , PatchFlag )

	out = false;

	PD = length(node);

	tol = 1e-4;

	if ( PD == 2 )

		switch PatchFlag

		case 'fullpatch'

			if ( (node(1)-Corners(1,1))<-tol || (node(2)-Corners(1,2))<-tol || (node(1)-Corners(2,1))>tol || (node(2)-Corners(4,2))>tol )

				out = true;

			end

		case 'vertpatch'

			if ( (node(2)-Corners(1,2))<-tol || (node(2)-Corners(4,2))>tol )

				out = true;

			end

		case 'horzpatch'

			if ( (node(1)-Corners(1,1))<-tol || (node(1)-Corners(2,1))>tol )

				out = true;

			end

		end

	elseif ( PD == 3 )

		switch PatchFlag

		case 'fullpatch'

			if ( (node(1)-Corners(1,1))<-tol || (node(2)-Corners(1,2))<-tol || (node(3)-Corners(1,3))<-tol || (node(1)-Corners(2,1))>tol || (node(2)-Corners(4,2))>tol || (node(3)-Corners(5,3))>tol )

				out = true;

			end

		case 'vertpatch'

			if ( (node(2)-Corners(1,2))<-tol || (node(2)-Corners(4,2))>tol )

				out = true;

			end

		case 'horzpatch'

			if ( (node(1)-Corners(1,1))<-tol || (node(1)-Corners(2,1))>tol )

				out = true;

			end

		end

	end

end

