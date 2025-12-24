function out = PostProcess( increment , PL , EL , Outcore , flag )

	% DON'T FORGET

	% you may need to create the folder "SIMULATIONS-LOCAL" in your home "~/" folder
	
	% POSTPROCESS function

	NoPs = size(PL,2);
	PD = PL(1).PD;

	switch flag

	case 'write-output-ParaView-points'

		filename = [Outcore '_PD_' '.' sprintf('%04d',increment) ,'.vtk'];

		file = fopen(filename,'wt');

		% header of vtk-file
		fprintf(file,'# vtk DataFile Version 3.1\n');
		fprintf(file,'THIS IS VTK OUTPUT\n');
		fprintf(file,'ASCII\n');
		fprintf(file, 'DATASET UNSTRUCTURED_GRID\n');

		% use field data to store time and cycle information

		dt = 1;

		fprintf(file, 'FIELD FieldData 2\n');
		fprintf(file,'TIME 1 1 double\n');
		fprintf(file, '%d\n', increment*dt);
		fprintf(file, 'CYCLE 1 1 int\n');
		fprintf(file, '%d\n',increment);

		% "points": node points

		fprintf(file, 'POINTS %d double\n', NoPs);

		for i=1:NoPs

			if ( PD == 2 )
			    fprintf(file,'%24.12e %24.12e %24.12e\n', [PL(i).X' 0]);
			elseif ( PD == 3 )    
			    fprintf(file,'%24.12e %24.12e %24.12e\n', PL(i).X');
			end

		end

		% " POINT_DATA": write SCALARS, VECTORS, TENSORS (in that order!)
		fprintf(file,'POINT_DATA %d\n',NoPs);

		% "SCALARS": 

		fprintf(file,'SCALARS %s double\n', 'NNbr');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',length(PL(i).neighbors));

		end

		fprintf(file,'SCALARS %s double\n', 'Vol');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).Vol);

		end

		fprintf(file,'SCALARS %s double\n', 'energy');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).energy);

		end

		fprintf(file,'SCALARS %s double\n', 'AV');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).AV);

		end

		fprintf(file,'SCALARS %s double\n', 'NI');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).NI);

		end

		fprintf(file,'SCALARS %s double\n', 'NInII');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).NInII);

		end

		fprintf(file,'SCALARS %s double\n', 'NInIInIII');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',PL(i).NInIInIII);

		end

		% write average point-wise volumes (length,area,volume)

		avg_vols = zeros(NoPs,6);

		for p=1:NoPs

			avg_vols(p,:) = PL(p).avg_volumes;

		end

		fprintf(file,'SCALARS %s double\n', 'L_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,1));

		end

		fprintf(file,'SCALARS %s double\n', 'A_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,2));

		end

		fprintf(file,'SCALARS %s double\n', 'V_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,3));

		end

		fprintf(file,'SCALARS %s double\n', 'l_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,4));

		end

		fprintf(file,'SCALARS %s double\n', 'a_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,5));

		end

		fprintf(file,'SCALARS %s double\n', 'v_avg');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,6));

		end

		% write average point-wise Jacobians

		fprintf(file,'SCALARS %s double\n', 'JavgI');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,4)/avg_vols(i,1));

		end

		fprintf(file,'SCALARS %s double\n', 'JavgInII');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoPs

		    fprintf(file,'%24.12e\n',avg_vols(i,5)/avg_vols(i,2));

		end

		if ( PD==3 )

			fprintf(file,'SCALARS %s double\n', 'JavgInIInIII');

		    fprintf(file, 'LOOKUP_TABLE default\n');

			for i=1:NoPs

			    fprintf(file,'%24.12e\n',avg_vols(i,6)/avg_vols(i,3));

			end

		end

		% VECTORS: write displacements

		fprintf(file,'VECTORS Disp double \n');

		for i=1:NoPs

			if ( PD == 2 )
			    fprintf(file,'%24.12e %24.12e %24.12e\n',[(PL(i).x'-PL(i).X') 0]);
			elseif ( PD == 3 )    
			    fprintf(file,'%24.12e %24.12e %24.12e\n', (PL(i).x'-PL(i).X'));
			end

		end

		fclose(file);

	case 'write-output-ParaView-elements'

		NoEs = size(EL,2);
		NPE = EL(1).NPE;

		NoNs = NoPs;

		for e = 1:NoEs

	        x = zeros(PD,NPE);

	        NdL = EL(e).NdL;

	        for i = 1:NPE
	            x(:,i) = PL(NdL(i)).x;
	        end

	        EL(e).x = x;

	    end

		filename = [Outcore '_PDFEBG_' '.' sprintf('%04d',increment) ,'.vtk'];

		file = fopen(filename,'wt');

		% header of vtk-file
		fprintf(file,'# vtk DataFile Version 3.1\n');
		fprintf(file,'THIS IS VTK OUTPUT\n');
		fprintf(file,'ASCII\n');
		fprintf(file, 'DATASET UNSTRUCTURED_GRID\n');

		% use field data to store time and cycle information

		dt = 1;

		fprintf(file, 'FIELD FieldData 2\n');
		fprintf(file,'TIME 1 1 double\n');
		fprintf(file, '%d\n', increment*dt);
		fprintf(file, 'CYCLE 1 1 int\n');
		fprintf(file, '%d\n',increment);

		if ( PD==2 && NPE==4 )
			vtk_element_indicator = 9; % Linear Tetragonal Element
			NPE_vtk = 4; % 4 nodes for Linear Tetragonal Element
		elseif ( PD==2 && NPE==9 )
			vtk_element_indicator = 23; % Quadratic Serendipity Tetragonal Element
			NPE_vtk = 8; % 8 nodes for Quadratic Serendipity Tetragonal Element
		elseif ( PD==3 && NPE==8 )
			vtk_element_indicator = 12; % Linear Hexahedral Element
			NPE_vtk = 8; % 8 nodes for Linear Hexahedral Element
		elseif ( PD==3 && NPE==27 )
			vtk_element_indicator = 25; % Quadratic Serendipity Hexahedral Element
			NPE_vtk = 20; % 20 nodes for Quadratic Serendipity Hexahedral Element
		end

		% "points": node points

		fprintf(file, 'POINTS %d double\n', NoNs);

		for i=1:NoNs

			if ( PD == 2 )
			    fprintf(file,'%24.12e %24.12e %24.12e\n',[PL(i).X' 0]);
			elseif ( PD == 3 )    
			    fprintf(file,'%24.12e %24.12e %24.12e\n', PL(i).X');
			end

		end

		% "CELLS": write element connectivity
		size_elem = 0;
		for i=1:NoEs
		    size_elem = size_elem + NPE_vtk + 1;
		end

		fprintf(file, 'CELLS %d %d\n', NoEs, size_elem );

		for i=1:NoEs 

		    fprintf(file, '%d ' , NPE_vtk);

		    NdL = EL(i).NdL;

		    if ( vtk_element_indicator == 9 || vtk_element_indicator == 23 || vtk_element_indicator == 12 )
		    	VTKNdL = NdL - 1; % vtk uses C arrays ... numbering in C starts with 0 not with 1
		    elseif ( vtk_element_indicator == 25 )
		    	VTKNdL(1:12) = NdL(1:12) - 1; % vtk uses C arrays ... numbering in C starts with 0 not with 1
		    	VTKNdL(13:16) = NdL(17:20) - 1; % vtk uses C arrays ... numbering in C starts with 0 not with 1
		    	VTKNdL(17:20) = NdL(13:16) - 1; % vtk uses C arrays ... numbering in C starts with 0 not with 1
		    end

		    for j=1:NPE_vtk
		        fprintf(file,' %8d', VTKNdL(j));
		    end

		    fprintf(file,'\n');
		    
		end

		% "CELL_TYPES": write type of each element (BAR2, QUAD4, etc.)
		% at the moment only quad4 is used -> vtkElType = 9
		fprintf(file,'CELL_TYPES %d\n', NoEs);

		for i=1:NoEs
			fprintf(file,'%d\n', vtk_element_indicator);
		end

		% " POINT_DATA": write SCALARS, VECTORS, TENSORS (in that order!)
		fprintf(file,'POINT_DATA %d\n',NoNs);

		% "SCALARS": no SCALARS implemented

		% VECTORS: write displacements

		fprintf(file,'VECTORS Disp double \n');

		for i=1:NoNs

			if ( PD == 2 )
			    fprintf(file,'%24.12e %24.12e %24.12e\n',[(PL(i).x'-PL(i).X') 0]);
			elseif ( PD == 3 )    
			    fprintf(file,'%24.12e %24.12e %24.12e\n', (PL(i).x'-PL(i).X'));
			end

		end

		%"CELL_DATA": write Scalars ..

		fprintf(file, 'CELL_DATA %d\n', NoEs);

		% %"SCALARS": write Material

	    fprintf(file,'SCALARS %s double\n', 'Mat');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoEs
	  
	        fprintf(file,'%24.12e\n',EL(i).Mat());

	    end

	    % %"SCALARS": write DetF

	    fprintf(file,'SCALARS %s double\n', 'Fdet');

	    fprintf(file, 'LOOKUP_TABLE default\n');

		for i=1:NoEs

		    [F,P,sigma] = EL(i).strss_and_more_single_GP();

	        fprintf(file,'%24.12e\n',det(F));

	    end

		% % "VECTORS": no VECTORS impelemented

		% % "TENSORS": P

		% fprintf(file,'TENSORS P double\n');

		% for i = 1:NoEs

		%     [F,P,sigma] = EL(i).strss_and_more_single_GP();

		%     if ( PD==2 )
		%         fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',[P(1,1) P(1,2) 0 ; P(2,1) P(2,2) 0 ; 0 0 0 ] );
		%     else
		%         fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',P);
		%     end

		% end

		fprintf(file,'TENSORS F double\n');

		for i = 1:NoEs

		    [F,P,sigma] = EL(i).strss_and_more_single_GP();

		    if ( PD==2 )
		        fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',[F(1,1) F(1,2) 0 ; F(2,1) F(2,2) 0 ; 0 0 1 ] );
		    else
		        fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',F);
		    end

		end

		% fprintf(file,'TENSORS sigma double\n');

		% for i = 1:NoEs

		%     [F,P,sigma] = EL(i).strss_and_more_single_GP();

		%     if ( PD==2 )
		%         fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',[sigma(1,1) sigma(1,2) 0 ; sigma(2,1) sigma(2,2) 0 ; 0 0 1 ] );
		%     else
		%         fprintf(file,'%24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n %24.12e %24.12e %24.12e\n',sigma);
		%     end

		% end

		fclose(file);

	case 'poisson'

		tol = 1e-6;

		delta_x = 0;
		delta_y = 0;

		L = PL(1).L;

		dd = 0.1;

		cntr = [0 0];

		for i=1:NoPs

			X = PL(i).X';

			if ( PD==2 )

				if ( abs(X(1)-cntr(1))<tol && abs(X(2)-dd)<tol )

					delta = PL(i).x'-PL(i).X';

					delta_y_1 = delta(2);

				elseif ( abs(X(1)-cntr(1))<tol && abs(X(2)+dd)<tol )

					delta = PL(i).x'-PL(i).X';

					delta_y_2 = delta(2);

				elseif ( abs(X(1)-dd)<tol && abs(X(2)-cntr(2))<tol )

					delta = PL(i).x'-PL(i).X';

					delta_x_1 = delta(1);

				elseif ( abs(X(1)+dd)<tol && abs(X(2)-cntr(2))<tol )

					delta = PL(i).x'-PL(i).X';

					delta_x_2 = delta(1);

				end

			end

		end

		delta_x = delta_x_1 - delta_x_2;
		delta_y = delta_y_1 - delta_y_2;

		nu = - delta_y/delta_x;

		out = nu;

	case 'write-output-graph-data'

		filename_AA = [Outcore '_PD_' 'graph_AA' , '.txt'];
		filename_BB = [Outcore '_PD_' 'graph_BB' , '.txt'];
		filename_CC = [Outcore '_PD_' 'graph_CC' , '.txt'];
		filename_DD = [Outcore '_PD_' 'graph_DD' , '.txt'];
		filename_EE = [Outcore '_PD_' 'graph_EE' , '.txt'];

		file_AA = fopen(filename_AA,'wt');
		file_BB = fopen(filename_BB,'wt');
		file_CC = fopen(filename_CC,'wt');
		file_DD = fopen(filename_DD,'wt');
		file_EE = fopen(filename_EE,'wt');

		tol = 1e-5;

		for i=1:NoPs

			X = PL(i).X;

			if ( abs(X(2)-0) < tol ) % AA

				fprintf(file_AA,'%24.12e %24.12e %24.12e %24.12e\n', [PL(i).X' (PL(i).x'-PL(i).X')]);

			elseif ( abs(X(2)-0.1) < tol ) % BB

				fprintf(file_BB,'%24.12e %24.12e %24.12e %24.12e\n', [PL(i).X' (PL(i).x'-PL(i).X')]);

			elseif ( abs(X(2)-0.2) < tol ) % CC

				fprintf(file_CC,'%24.12e %24.12e %24.12e %24.12e\n', [PL(i).X' (PL(i).x'-PL(i).X')]);

			elseif ( abs(X(2)-0.4) < tol ) % DD

				fprintf(file_DD,'%24.12e %24.12e %24.12e %24.12e\n', [PL(i).X' (PL(i).x'-PL(i).X')]);

			elseif ( abs(X(2)-0.5) < tol ) % EE

				fprintf(file_EE,'%24.12e %24.12e %24.12e %24.12e\n', [PL(i).X' (PL(i).x'-PL(i).X')]);

			end

		end

		fclose(file_AA);
		fclose(file_BB);
		fclose(file_CC);
		fclose(file_DD);
		fclose(file_EE);

	end

end
