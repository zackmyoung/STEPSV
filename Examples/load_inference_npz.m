function [t, p, pstd, lo1, hi1, lo2, hi2] = load_inference_npz(npz_file, varargin)
% [t, p, pstd, lo1, hi1, lo2, hi2] = load_inference_npz(npz_file, 'Sigma', 1)
%
% Loads arrays from NPZ: time, prob, optionally prob_std.
% Returns:
%   t    : decimal years (column)
%   p    : probability (column)
%   pstd : std dev (column) or []
%   lo1/hi1 : p +/- 1*sigma bounds (clipped to [0,1]) if std exists, else []
%   lo2/hi2 : p +/- 2*sigma bounds (clipped) if std exists, else []
%
% Options:
%   'Sigma' (default 1): base sigma multiplier

pIn = inputParser;
addParameter(pIn, 'Sigma', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
parse(pIn, varargin{:});
sigma = pIn.Results.Sigma;

d = read_npz(npz_file);

t = d.time(:);
p = d.prob(:);

pstd = [];
lo1 = []; hi1 = []; lo2 = []; hi2 = [];

if isfield(d, 'prob_std')
    pstd = d.prob_std(:);

    lo1 = max(0, p - sigma*pstd);
    hi1 = min(1, p + sigma*pstd);

    lo2 = max(0, p - 2*sigma*pstd);
    hi2 = min(1, p + 2*sigma*pstd);
end
end


function plot_inference_npz(npz_file, varargin)
% plot_inference_npz(npz_file, 'Sigma', 1, 'Show2Sigma', true)

pIn = inputParser;
addParameter(pIn, 'Sigma', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(pIn, 'Show2Sigma', false, @(x) islogical(x) || isnumeric(x));
parse(pIn, varargin{:});
sigma = pIn.Results.Sigma;
show2 = logical(pIn.Results.Show2Sigma);

[t, p, pstd, lo1, hi1, lo2, hi2] = load_inference_npz(npz_file, 'Sigma', sigma);

figure('Color','w'); hold on; grid on;

if ~isempty(pstd)
    % 2-sigma first (wider, lighter)
    if show2
        fill([t; flipud(t)], [lo2; flipud(hi2)], [0.75 0.75 0.75], ...
            'EdgeColor','none', 'FaceAlpha', 0.25);
    end
    % 1-sigma (narrower, darker)
    fill([t; flipud(t)], [lo1; flipud(hi1)], [0.55 0.55 0.55], ...
        'EdgeColor','none', 'FaceAlpha', 0.35);
end

plot(t, p, 'k-', 'LineWidth', 1.2);
ylim([-0.05 1.05]);
xlabel('Decimal year');
ylabel('P(step)');
title(strrep(npz_file, '_', '\_'));
end


function out = read_npz(npz_file)
% Minimal NPZ reader: extracts npy files to temp folder and reads them.
tmpdir = fullfile(tempdir, ['npz_', char(java.util.UUID.randomUUID)]);
mkdir(tmpdir);

unzip(npz_file, tmpdir);
files = dir(fullfile(tmpdir, '*.npy'));

out = struct();
for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    [~, name, ~] = fileparts(files(k).name);
    out.(name) = read_npy(fn);
end

try rmdir(tmpdir, 's'); catch, end
end


function A = read_npy(npy_file)
% Basic NPY reader for little-endian numeric arrays.
fid = fopen(npy_file, 'r');
assert(fid>0, 'Could not open %s', npy_file);
cleanup = onCleanup(@() fclose(fid));

magic = fread(fid, 6, '*char')';
assert(strcmp(magic, char([147 'NUMPY'])), 'Not an NPY file: %s', npy_file);

fread(fid, 2, 'uint8'); % version
hlen = fread(fid, 1, 'uint16');
hdr = fread(fid, double(hlen), '*char')';
hdr = string(hdr);

descr = extractBetween(hdr, "'descr': '", "'");
descr = char(descr);
assert(~isempty(descr), 'Could not parse dtype.');

endi = descr(1);
typ  = descr(2);
bytes = str2double(descr(3:end));

fortran = contains(hdr, "'fortran_order': True");

shape_txt = extractBetween(hdr, "'shape': (", ")");
shape_txt = char(shape_txt);
shape_nums = regexp(shape_txt, '\d+', 'match');
shape = str2double(shape_nums);
if isempty(shape), shape = 1; end

if typ == 'f'
    prec = tern(bytes==8, 'double', 'single');
elseif typ == 'i'
    prec = sprintf('int%d', bytes*8);
elseif typ == 'u'
    prec = sprintf('uint%d', bytes*8);
else
    error('Unsupported dtype "%s"', descr);
end

if endi == '>'
    error('Big-endian NPY not supported in this minimal reader.');
end

A = fread(fid, prod(shape), ['*' prec]);

if numel(shape) > 1
    A = reshape(A, shape);
else
    A = reshape(A, [shape 1]);
end

if ~fortran && numel(shape) > 1
    A = permute(A, numel(shape):-1:1);
end
end


function y = tern(cond, a, b)
if cond, y = a; else, y = b; end
end
