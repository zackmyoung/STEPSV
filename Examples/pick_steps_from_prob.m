function step_times = pick_steps_from_prob(t, p, varargin)
% step_times = pick_steps_from_prob(t, p, 'Thresh',0.6,'MinSepDays',14,'IgnoreFirstN',30,'MaxSteps',20)

ip = inputParser;
addParameter(ip, 'Thresh', 0.6, @(x) isnumeric(x) && isscalar(x));
addParameter(ip, 'MinSepDays', 14, @(x) isnumeric(x) && isscalar(x) && x>=0);
addParameter(ip, 'IgnoreFirstN', 0, @(x) isnumeric(x) && isscalar(x) && x>=0);
addParameter(ip, 'MaxSteps', 20, @(x) isnumeric(x) && isscalar(x) && x>=1);
parse(ip, varargin{:});

th       = ip.Results.Thresh;
minSepYrs= ip.Results.MinSepDays / 365.25;
ign      = ip.Results.IgnoreFirstN;
maxSteps = ip.Results.MaxSteps;

t = t(:); p = p(:);
pp = p;

% Ignore first N samples (to avoid start-of-series artifacts)
if ign > 0
    pp(1:min(ign, numel(pp))) = -Inf;  % ensures they are never selected as peaks
end

% Use MATLAB's built-in peak picker (Signal Processing Toolbox)
% - MinPeakHeight enforces your threshold
% - MinPeakDistance enforces your min separation (in samples)
dt_yrs = median(diff(t));
if ~isfinite(dt_yrs) || dt_yrs <= 0
    step_times = [];
    return;
end
minDistSamp = max(1, round(minSepYrs / dt_yrs));


% ✅ robust guard
pp_f = pp(isfinite(pp));
if isempty(pp_f) || max(pp_f) <= th
    step_times = [];
    return;
end

[pk, loc] = findpeaks(pp, 'MinPeakHeight', th, 'MinPeakDistance', minDistSamp);

if isempty(loc)
    step_times = [];
    return;
end

% Keep at most MaxSteps, preferring largest peaks (like your old sort-by-prob)
[~, ord] = sort(pk, 'descend');
loc = loc(ord);
loc = loc(1:min(maxSteps, numel(loc)));

step_times = sort(t(loc));
end
