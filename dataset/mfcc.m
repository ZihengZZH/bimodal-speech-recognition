base = double('A') - 1;
names = {"Anya", "Bill", "Faye", "John", "Kate", "Nicola", "Stephen", "Steve", "Verity", "Yi"};
for i = 1:26
  for j = 1:10
    for k = 1:3
      name = sprintf('%s%d_%s.mfcc', char(base+i), k, names{j});
      mfccfile = fopen(strcat('avletters/Audio/mfcc/Clean/', name), 'r', 'b');
      nSamples = fread(mfccfile, 1, 'int32');
      sampPeriod = fread(mfccfile, 1, 'int32') * 1E-7;
      sampSize = 0.25 * fread(mfccfile, 1, 'int16');
      parmKind = fread(mfccfile, 1, 'int16');
      features = fread(mfccfile, [sampSize, nSamples], 'float' ).';
      output_name = sprintf('%s%d_%s.mat', char(base+i), k, names{j});
      save(strcat('processed/avletters/mfccs/', output_name), 'features');
      fclose(mfccfile);
    end
  end
end


