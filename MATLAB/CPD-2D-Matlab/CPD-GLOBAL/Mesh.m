function [ NL , CNCT ] = Mesh( Corners , L )

	PD = size(Corners(1,:),2);

	% generate NodeList

	if ( PD == 2 )

		A = Corners(1,:); % left south corner
		B = Corners(2,:); % right south corner
		D = Corners(4,:); % left north corner

		Nx = round( (B(1)-A(1))/L )+1; % number of nodes along x including A and B
		Ny = round( (D(2)-A(2))/L )+1; % number of nodes along y including A and D

		xx = linspace(A(1),B(1),Nx); % xx includes A and B
		yy = linspace(A(2),D(2),Ny); % yy includes A and D

		NoNs = Nx*Ny;

		NL = zeros(NoNs,PD);

		for j = 1:Ny

			for i = 1:Nx

				n = (j-1)*Nx + i;

				NL(n,:) = [ xx(i) , yy(j) ];

			end

		end

	elseif ( PD == 3 )

		A = Corners(1,:); % left south bottom corner
		B = Corners(2,:); % right south bottom corner
		D = Corners(4,:); % right north bottom corner
		E = Corners(5,:); % left north top corner

		Nx = round( (B(1)-A(1))/L )+1; % number of nodes along x including A and B
		Ny = round( (D(2)-A(2))/L )+1; % number of nodes along y including A and D
		Nz = round( (E(3)-A(3))/L )+1; % number of nodes along z including A and E

		xx = linspace(A(1),B(1),Nx); % xx includes A and B
		yy = linspace(A(2),D(2),Ny); % yy includes A and D
		zz = linspace(A(3),E(3),Nz); % zz includes A and E

		NoNs = Nx*Ny*Nz;

		NL = zeros(NoNs,PD);

		for k = 1:Nz

			for j = 1:Ny

				for i = 1:Nx

					n = (k-1)*Ny*Nx + (j-1)*Nx + i;

					NL(n,:) = [ xx(i) , yy(j) , zz(k) ];

				end

			end

		end

	end	

	% generate Connectivity

	if ( nargout==2 )

		if ( PD == 2 )

			NPE = 4;
			CNCT = zeros((Ny-1)*(Nx-1),NPE);

			for j = 1:Ny-1

				for i = 1:Nx-1

					e = (j-1)*(Nx-1) + i;

					CNCT(e,1) = (j-1)*Nx + i;
					CNCT(e,2) = (j-1)*Nx + i+1;
					CNCT(e,3) = j*Nx + i+1;
					CNCT(e,4) = j*Nx + i;

				end

			end

		elseif ( PD == 3 )

			NPE = 8;
			CNCT = zeros((Nz-1)*(Ny-1)*(Nx-1),NPE);

			for k = 1:Nz-1

				for j = 1:Ny-1

					for i = 1:Nx-1

						e = (k-1)*(Ny-1)*(Nx-1) + (j-1)*(Nx-1) + i;

						CNCT(e,1) = (k-1)*Ny*Nx + (j-1)*Nx + i;
						CNCT(e,2) = (k-1)*Ny*Nx + (j-1)*Nx + i+1;
						CNCT(e,3) = (k-1)*Ny*Nx + j*Nx + i+1;
						CNCT(e,4) = (k-1)*Ny*Nx + j*Nx + i;
						CNCT(e,5) = k*Ny*Nx + (j-1)*Nx + i;
						CNCT(e,6) = k*Ny*Nx + (j-1)*Nx + i+1;
						CNCT(e,7) = k*Ny*Nx + j*Nx + i+1;
						CNCT(e,8) = k*Ny*Nx + j*Nx + i;

					end

				end

			end

		end

	end

end
