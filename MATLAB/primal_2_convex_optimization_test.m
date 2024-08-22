% PRIMAL PROBLEM 2.0
% Uses rescaling, intermediate points, and full subtraction 

% Running problem 2 using interpolation points.
% Prepared to run offline (remove plots)

% M is the number of points on the half circle
% saXXX is the subtractions point s0 = XXX

% st1 = ["-1p00", "0p00", "1p33", "2p00", "3p00", "3p80"];
% st1_python = ["-1", "0", "1p33", "2", "3", "3p8"];

st1 = ["2p00"];
ct1 = ["ko", "go", "ko", "co", "yo"];
Mv = [40];
lmaxv = [20];

mreg = [8];
nfv = [1];
nv = [50];

% mreg = [-4:0.2:10];
% mreg = [2 4 6 8];
% mreg = [-4 0 2 4 7 9];  % regulator
% nfv = [1 7 9 13];       % ntheta
% nv = [4:2:20];          % ncoeffs Requires having M from previous run
% tv = zeros(length(st1), length(mreg));
% alphav = [0:0.1:2*pi];
% tv = zeros(length(st1), length(mreg) * length(nv) * length(nfv));

alphav = [0:0.001:2*pi];
tv  = zeros(length(st1), length(mreg), length(alphav));
c0v = zeros(length(st1), length(mreg), length(alphav));
c2v = zeros(length(st1), length(mreg), length(alphav));
fdv = zeros(size(mreg));


for jcount = 1:length(st1)
    M = Mv(jcount);
    lmax = lmaxv(jcount);

    % testFile = sprintf('test.txt');
    % testID = fopen(testFile, 'r');
    % test = fscanf(testID, '%f', [16 2]);
    % disp(test);
    
    % dataKFile = sprintf('primal_2.0_points_data_P2_nopole_sa%s_lmax%d_M%d.dat', st1(jcount), lmax, M);
    % dataOutFile = sprintf('primal_2.0_points_data_P2_nopole_sa%s_lmax%d_M%d_ncoeff%d_outB.datx', st1(jcount), lmax, M, max(nv));
    dataKFile = sprintf('dual_1.0_points_data_P2_nopole_sa%s_lmax%d_M%d.dat', st1(jcount), lmax, M);
    dataOutFile = sprintf('dual_1.0_points_data_P2_nopole_sa%s_lmax%d_M%d_ncoeff%d_outB.datx', st1(jcount), lmax, M, max(nv));

    Mfl = (lmax + 1) * M;
    Mfn = M * (M + 1) / 2;
    Mfn2 = (M + 1) * (M + 2) / 2;
    
    fileID = fopen(dataKFile, 'r');
    xiv = fscanf(fileID, '%f', [M]);

    % REGULAR
    a0h  = fscanf(fileID, '%f', 1);
    b0h  = fscanf(fileID, '%f', 1);
    h0R  = fscanf(fileID, '%f', [Mfl]);
    a1h  = fscanf(fileID, '%f', [M]);
    b1h  = fscanf(fileID, '%f', [M]);
    h1R  = fscanf(fileID, '%f', [Mfl M]);
    h1I  = fscanf(fileID, '%f', [Mfl M]);
    a2h  = fscanf(fileID, '%f', [Mfn]);
    b2h  = fscanf(fileID, '%f', [Mfn]);
    h2R  = fscanf(fileID, '%f', [Mfl Mfn]);
    h2I  = fscanf(fileID, '%f', [Mfl Mfn]);
    % disp(size(a0h));
    % disp(size(b0h));
    % disp(size(h0R));
    % disp(size(a1h));
    % disp(size(b1h));
    % disp(size(h1R));
    % disp(size(h1I));
    % disp(size(a2h));
    % disp(size(b2h));
    % disp(size(h2R));
    % disp(size(h2I));

    % RESCALED
    h0RA  = fscanf(fileID, '%f', [Mfl]);
    h1RA  = fscanf(fileID, '%f', [Mfl M]);
    h1IA  = fscanf(fileID, '%f', [Mfl M]);
    h2RA  = fscanf(fileID, '%f', [Mfl Mfn]);
    h2IA  = fscanf(fileID, '%f', [Mfl Mfn]);
    h1IB  = fscanf(fileID, '%f', [Mfl M]);
    h2IB  = fscanf(fileID, '%f', [Mfl Mfn]);
    Lambdav = fscanf(fileID, '%f', [Mfl]);

    % disp(size(h0RA));
    % disp(h1RA);
    % disp(size(h1IA));
    % disp(size(h2RA));
    % disp(size(h2IA));
    % disp(size(h1IB));
    % disp(size(h2IB));

    fclose(fileID);

    % REORDER AND RECONDITION LINEAR CONSTRAINT MATRIX
    A = [[h0RA h1RA h2RA]; [zeros(Mfl, 1) h1IA h2IA]; [zeros(Mfl, 1) h1IB h2IB]]; % with rescaling 
    % disp(A(1:7, 1:7));
    % disp(size(A));

    [Q, R, P] = qr(A', 'vector');
    % disp(Q(1:7, 1:7));
    % disp(R(1:7, 1:7));
    % disp(P(1:7));

    ap = [a0h a1h' a2h'] * Q;
    bp = [b0h b1h' b2h'] * Q;
    % disp(ap(1:7));
    % disp(size(ap));
    % disp(bp(1:7));
    % disp(size(bp));

    
    invP(P) = 1:length(P);
    disp(invP(1:7));
    indRA = invP(1:Mfl);
    indIA = invP(Mfl+1:2*Mfl);
    indIB = invP(2*Mfl+1:3*Mfl);
    disp(indRA(1:7));
    disp(indIA(1:7));
    disp(indIB(1:7));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fileID = fopen(dataOutFile, 'w');
    
    sv = 4./cos(xiv/2).^2;
    count_alpha = 1;
    
    for alpha = alphav
        count=1;
        for nf = nfv
            for n=nv  
                for Mreg = 10.^mreg
                    % print(Mreg)
                    count=count+1;
     
                end % REGULATOR mreg
            end % NUMBER OF COEFFICIENTS ncoeff
        end % NUMBER OF REGULATED COMPONENTS ntheta

    count_alpha = count_alpha + 1;
    end % ALPHA ANGLE IN c0 c2-plane

    fclose(fileID);
end % SUBTRACTION POINT

