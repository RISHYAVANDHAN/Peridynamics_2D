function [ PL , EL ] = Topology( NL , CNCT , L , Delta , Bvals , TOPflag )

	NoNs = size(NL,1);
	PD = size(NL,2);

	NoP = 0;        % point number

	NPL = zeros(1,NoNs);
	
	switch TOPflag

    case 'FULL'

		for n = 1 : NoNs

			X = NL(n,:);

			NoP = NoP+1;

			PL(NoP) = Point(NoP,X');

			PL(NoP).NNr = n;

			NPL(n) = NoP;

		end

	case 'PART'

		if ( PD==2 )

			R = Bvals(1);
			Xo = Bvals(2);
			Yo = Bvals(3);

			tol = 1e-3;

			for n = 1 : NoNs

				X = NL(n,:);
				
				if ( ( (X(1) - Xo)^2 + (X(2) - Yo)^2 - R^2 ) > -tol )

					NoP = NoP+1;

					PL(NoP) = Point(NoP,X');

					PL(NoP).NNr = n;

					NPL(n) = NoP;

				end

			end

		elseif ( PD==3 )

			R = Bvals(1);
			Xo = Bvals(2);
			Yo = Bvals(3);
			Zo = Bvals(4);

			tol = 1e-3;

			for n = 1 : NoNs

				X = NL(n,:);
				
				if ( ( (X(1) - Xo)^2 + (X(2) - Yo)^2 + (X(3) - Zo)^2 - R^2 ) > -tol )

					NoP = NoP+1;

					PL(NoP) = Point(NoP,X');

					PL(NoP).NNr = n;

					NPL(n) = NoP;

				end

			end

		end

	case 'CUT'

		% v = round(Delta/L)*L;
		v = 0; % if non-zero, we will have additional layers of points

		if ( PD==2 )

			R = Bvals(1)-v;
			Xo = Bvals(2);
			Yo = Bvals(3);

			tol = 1e-3;

			for n = 1 : NoNs

				X = NL(n,:);
				
				if ( (X(1) - (Xo+R))>-tol || (X(1) - (Xo-R))<tol || (X(2) - (Yo+R))>-tol || (X(2) - (Yo-R))<tol )

					NoP = NoP+1;

					PL(NoP) = Point(NoP,X');

					PL(NoP).NNr = n;

					NPL(n) = NoP;

				end

			end

		elseif ( PD==3 )

			R = Bvals(1)-v;
			Xo = Bvals(2);
			Yo = Bvals(3);
			Zo = Bvals(4);

			tol = 1e-3;

			for n = 1 : NoNs

				X = NL(n,:);
				
				if ( (X(1) - (Xo+R))>-tol || (X(1) - (Xo-R))<tol || (X(2) - (Yo+R))>-tol || (X(2) - (Yo-R))<tol || (X(3) - (Zo+R))>-tol || (X(3) - (Zo-R))<tol )

					NoP = NoP+1;

					PL(NoP) = Point(NoP,X');

					PL(NoP).NNr = n;

					NPL(n) = NoP;

				end

			end

		end

	end % switch case TOPflag

	if ( nargout==2 )

		NoE = 0;        % element number

		NoEs = size(CNCT,1);
		NPE = size(CNCT,2);

		switch TOPflag

	    case 'FULL'

	    	PNdL = zeros(1,NPE);

	    	for e = 1 : NoEs

	    		NdL = CNCT(e,:);

	    		for i = 1 : NPE

	            	XX(:,i) = (NL(NdL(i),:))';

	            	PNdL(i) = NPL(NdL(i));

	            end

				NoE = NoE+1;

	            E = Element(NoE,PNdL,XX);

	            EL(NoE) = E;

			end

		case 'PART'

			tol = 1e-3;

			PNdL = zeros(1,NPE);

	    	for e = 1 : NoEs

	    		NdL = CNCT(e,:);

	    		for i = 1 : NPE

	            	XX(:,i) = (NL(NdL(i),:))';

	            	PNdL(i) = NPL(NdL(i));

	            end

	            if ( PD==2 )

	            	R = Bvals(1);
					Xo = Bvals(2);
					Yo = Bvals(3);

	            	if ( ( (XX(1,1) - Xo)^2 + (XX(2,1) - Yo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,2) - Xo)^2 + (XX(2,2) - Yo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,3) - Xo)^2 + (XX(2,3) - Yo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,4) - Xo)^2 + (XX(2,4) - Yo)^2 - R^2 ) > -tol )

		            	NoE = NoE+1;

			            E = Element(NoE,PNdL,XX);

			            EL(NoE) = E;

					end

				elseif ( PD==3 )

					R = Bvals(1);
					Xo = Bvals(2);
					Yo = Bvals(3);
					Zo = Bvals(4);

					if ( ( (XX(1,1) - Xo)^2 + (XX(2,1) - Yo)^2 + (XX(3,1) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,2) - Xo)^2 + (XX(2,2) - Yo)^2 + (XX(3,2) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,3) - Xo)^2 + (XX(2,3) - Yo)^2 + (XX(3,3) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,4) - Xo)^2 + (XX(2,4) - Yo)^2 + (XX(3,4) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,5) - Xo)^2 + (XX(2,5) - Yo)^2 + (XX(3,5) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,6) - Xo)^2 + (XX(2,6) - Yo)^2 + (XX(3,6) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,7) - Xo)^2 + (XX(2,7) - Yo)^2 + (XX(3,7) - Zo)^2 - R^2 ) > -tol && ...
	            		 ( (XX(1,8) - Xo)^2 + (XX(2,8) - Yo)^2 + (XX(3,8) - Zo)^2 - R^2 ) > -tol )

		            	NoE = NoE+1;

			            E = Element(NoE,PNdL,XX);

			            EL(NoE) = E;

					end

				end

			end

		case 'CUT'

			tol = 1e-3;

			PNdL = zeros(1,NPE);

	    	for e = 1 : NoEs

	    		NdL = CNCT(e,:);

	    		for i = 1 : NPE

	            	XX(:,i) = (NL(NdL(i),:))';

	            	PNdL(i) = NPL(NdL(i));

	            end

	            if ( PD==2 )

	            	R = Bvals(1);
					Xo = Bvals(2);
					Yo = Bvals(3);

	            	if ( ( (XX(1,1) - (Xo+R))>-tol || (XX(1,1) - (Xo-R))<tol || (XX(2,1) - (Yo+R))>-tol || (XX(2,1) - (Yo-R))<tol ) && ...
	            		 ( (XX(1,2) - (Xo+R))>-tol || (XX(1,2) - (Xo-R))<tol || (XX(2,2) - (Yo+R))>-tol || (XX(2,2) - (Yo-R))<tol ) && ...
	            		 ( (XX(1,3) - (Xo+R))>-tol || (XX(1,3) - (Xo-R))<tol || (XX(2,3) - (Yo+R))>-tol || (XX(2,3) - (Yo-R))<tol ) && ...
	            		 ( (XX(1,4) - (Xo+R))>-tol || (XX(1,4) - (Xo-R))<tol || (XX(2,4) - (Yo+R))>-tol || (XX(2,4) - (Yo-R))<tol ) )

		            	NoE = NoE+1;

			            E = Element(NoE,PNdL,XX);

			            EL(NoE) = E;

					end

				elseif ( PD==3 )

					R = Bvals(1);
					Xo = Bvals(2);
					Yo = Bvals(3);
					Zo = Bvals(4);

					if ( ( (XX(1,1) - (Xo+R))>-tol || (XX(1,1) - (Xo-R))<tol || (XX(2,1) - (Yo+R))>-tol || (XX(2,1) - (Yo-R))<tol || (XX(3,1) - (Zo+R))>-tol || (XX(3,1) - (Zo-R))<tol ) && ... 
						 ( (XX(1,2) - (Xo+R))>-tol || (XX(1,2) - (Xo-R))<tol || (XX(2,2) - (Yo+R))>-tol || (XX(2,2) - (Yo-R))<tol || (XX(3,2) - (Zo+R))>-tol || (XX(3,2) - (Zo-R))<tol ) && ... 
						 ( (XX(1,3) - (Xo+R))>-tol || (XX(1,3) - (Xo-R))<tol || (XX(2,3) - (Yo+R))>-tol || (XX(2,3) - (Yo-R))<tol || (XX(3,3) - (Zo+R))>-tol || (XX(3,3) - (Zo-R))<tol ) && ... 
						 ( (XX(1,4) - (Xo+R))>-tol || (XX(1,4) - (Xo-R))<tol || (XX(2,4) - (Yo+R))>-tol || (XX(2,4) - (Yo-R))<tol || (XX(3,4) - (Zo+R))>-tol || (XX(3,4) - (Zo-R))<tol ) && ... 
						 ( (XX(1,5) - (Xo+R))>-tol || (XX(1,5) - (Xo-R))<tol || (XX(2,5) - (Yo+R))>-tol || (XX(2,5) - (Yo-R))<tol || (XX(3,5) - (Zo+R))>-tol || (XX(3,5) - (Zo-R))<tol ) && ... 
						 ( (XX(1,6) - (Xo+R))>-tol || (XX(1,6) - (Xo-R))<tol || (XX(2,6) - (Yo+R))>-tol || (XX(2,6) - (Yo-R))<tol || (XX(3,6) - (Zo+R))>-tol || (XX(3,6) - (Zo-R))<tol ) && ... 
						 ( (XX(1,7) - (Xo+R))>-tol || (XX(1,7) - (Xo-R))<tol || (XX(2,7) - (Yo+R))>-tol || (XX(2,7) - (Yo-R))<tol || (XX(3,7) - (Zo+R))>-tol || (XX(3,7) - (Zo-R))<tol ) && ... 
						 ( (XX(1,8) - (Xo+R))>-tol || (XX(1,8) - (Xo-R))<tol || (XX(2,8) - (Yo+R))>-tol || (XX(2,8) - (Yo-R))<tol || (XX(3,8) - (Zo+R))>-tol || (XX(3,8) - (Zo-R))<tol ) )

		            	NoE = NoE+1;

			            E = Element(NoE,PNdL,XX);

			            EL(NoE) = E;

					end

				end

			end

		end % switch case TOPflag

	end

end
