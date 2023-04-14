import os

import numpy as np
import pytest
import torch

import ocf_datapipes
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from torch.profiler import profile, record_function, ProfilerActivity

"""
After running it 50 times in a row, this is the breakdown, creating the sun image is by far the largest single event
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 enumerate(DataPipe)#ZipperIterDataPipe         0.13%     569.522ms       222.75%     1011.363s     777.972ms          1300  
                enumerate(DataPipe)#BatcherIterDataPipe         0.00%      16.264ms        99.89%      453.534s        3.024s           150  
            enumerate(DataPipe)#StackXarrayIterDataPipe         0.06%     252.146ms        98.66%      447.931s        8.959s            50  
         enumerate(DataPipe)#CreateSunImageIterDataPipe        75.59%      343.199s        75.59%      343.201s        6.864s            50  
                     enumerate(DataPipe)#_ChildDataPipe         0.07%     340.487ms        24.25%      110.086s      45.869ms          2400  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe         9.35%       42.436s        16.60%       75.387s     376.933ms           200  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         1.19%        5.414s         9.39%       42.636s     426.362ms           100  
              enumerate(DataPipe)#NormalizeIterDataPipe         0.83%        3.786s         7.48%       33.960s      75.467ms           450  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         0.82%        3.731s         7.26%       32.951s      82.378ms           400  
        enumerate(DataPipe)#SelectTimeSliceIterDataPipe         0.12%     524.390ms         6.44%       29.251s      83.574ms           350  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 454.026s

Same without CreateSunImage being used, Stack Xarray is somewhat required I think
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 enumerate(DataPipe)#ZipperIterDataPipe         0.55%     814.140ms       285.56%      424.116s     326.243ms          1300  
                enumerate(DataPipe)#BatcherIterDataPipe         0.01%      19.418ms        99.55%      147.851s     985.671ms           150  
            enumerate(DataPipe)#StackXarrayIterDataPipe         0.17%     248.443ms        95.31%      141.558s        2.831s            50  
                     enumerate(DataPipe)#_ChildDataPipe         0.30%     439.869ms        93.55%      138.936s      59.122ms          2350  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe        42.37%       62.922s        72.75%      108.043s     540.217ms           200  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         4.35%        6.468s        33.95%       50.418s     504.176ms           100  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         3.48%        5.174s        30.38%       45.122s     112.804ms           400  
              enumerate(DataPipe)#NormalizeIterDataPipe         3.33%        4.939s        26.57%       39.466s      87.703ms           450  
                 enumerate(DataPipe)#MapperIterDataPipe        18.72%       27.802s        23.92%       35.527s     177.633ms           200  
        enumerate(DataPipe)#SelectTimeSliceIterDataPipe         0.47%     692.004ms        22.44%       33.328s      95.223ms           350  
enumerate(DataPipe)#AddT0IdxAndSamplePeriodDurationI...         0.07%      98.052ms        18.34%       27.244s      68.110ms           400  
enumerate(DataPipe)#SelectTrainTestTimePeriodsIterDa...         0.11%     158.366ms        14.45%       21.465s     214.653ms           100  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe        14.33%       21.287s        14.33%       21.287s     212.866ms           100  
           enumerate(DataPipe)#SelectT0TimeIterDataPipe         0.03%      49.333ms         7.38%       10.966s     109.659ms           100  
      enumerate(DataPipe)#SelectTimePeriodsIterDataPipe         0.39%     581.290ms         7.35%       10.917s     109.166ms           100  
enumerate(DataPipe)#SelectOverlappingTimeSliceIterDa...         1.43%        2.123s         6.82%       10.129s     101.292ms           100  
enumerate(DataPipe)#GetContiguousT0TimePeriodsIterDa...         1.32%        1.964s         5.32%        7.898s      19.744ms           400  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         2.80%        4.164s         2.80%        4.164s      20.822ms           200  
         enumerate(DataPipe)#SelectChannelsIterDataPipe         0.29%     424.060ms         2.51%        3.733s      18.666ms           200  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         2.07%        3.074s         2.07%        3.082s      61.633ms            50  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 148.522s

Sorting by self_cpu_time_total, no HRV or Sun but everything else
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe        39.43%       37.949s        68.76%       66.180s     441.200ms           150  
                 enumerate(DataPipe)#MapperIterDataPipe        20.42%       19.650s        27.09%       26.073s     130.366ms           200  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe        17.73%       17.062s        17.73%       17.062s     170.617ms           100  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         5.59%        5.380s        33.95%       32.679s     326.794ms           100  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         2.70%        2.603s         2.71%        2.608s      52.168ms            50  
              enumerate(DataPipe)#NormalizeIterDataPipe         2.26%        2.173s        27.18%       26.160s      74.744ms           350  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         2.09%        2.011s        29.33%       28.231s      94.103ms           300  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         1.56%        1.500s         1.56%        1.500s      14.999ms           100  
         enumerate(DataPipe)#OpenTopographyIterDataPipe         1.29%        1.244s         1.29%        1.244s      12.438ms           100  
enumerate(DataPipe)#GetContiguousT0TimePeriodsIterDa...         1.10%        1.054s         4.01%        3.862s      12.872ms           300  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 96.247s

Same as above, but no Sat, just HRV instead

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe        43.58%       43.977s        71.71%       72.370s     482.469ms           150  
                 enumerate(DataPipe)#MapperIterDataPipe        19.35%       19.530s        25.65%       25.885s     129.426ms           200  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe        15.90%       16.044s        15.90%       16.044s     160.436ms           100  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         5.18%        5.228s        34.62%       34.933s     349.332ms           100  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         2.53%        2.552s         2.53%        2.557s      51.142ms            50  
              enumerate(DataPipe)#NormalizeIterDataPipe         2.33%        2.347s        24.81%       25.037s      71.533ms           350  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         2.06%        2.082s        28.14%       28.394s      94.646ms           300  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         1.39%        1.406s         1.39%        1.406s      14.061ms           100  
         enumerate(DataPipe)#OpenTopographyIterDataPipe         1.24%        1.252s         1.24%        1.252s      12.521ms           100  
enumerate(DataPipe)#GetContiguousT0TimePeriodsIterDa...         1.05%        1.057s         3.58%        3.612s      12.040ms           300  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 100.917s

Both sallites, no NWP
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 enumerate(DataPipe)#MapperIterDataPipe        28.64%       19.646s        37.98%       26.054s     130.272ms           200  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe        24.58%       16.861s        24.58%       16.861s     168.606ms           100  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe        12.46%        8.544s        55.70%       38.209s     254.729ms           150  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         8.20%        5.625s        55.00%       37.726s     377.262ms           100  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         5.29%        3.632s        43.24%       29.665s      98.883ms           300  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         4.40%        3.018s         4.40%        3.018s      15.088ms           200  
              enumerate(DataPipe)#NormalizeIterDataPipe         3.69%        2.531s        38.14%       26.161s      74.746ms           350  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         3.56%        2.439s         3.56%        2.445s      48.903ms            50  
         enumerate(DataPipe)#OpenTopographyIterDataPipe         1.74%        1.196s         1.74%        1.196s      11.965ms           100  
enumerate(DataPipe)#SelectOverlappingTimeSliceIterDa...         1.54%        1.057s         7.79%        5.345s      53.449ms           100  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 68.599s

Satellites plus Sun, no NWP
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         enumerate(DataPipe)#CreateSunImageIterDataPipe        81.27%      346.180s        81.27%      346.181s        6.924s            50  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe         5.37%       22.865s         5.37%       22.865s     228.648ms           100  
                 enumerate(DataPipe)#MapperIterDataPipe         3.53%       15.026s         5.29%       22.530s     112.651ms           200  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe         2.67%       11.393s         9.34%       39.781s     265.205ms           150  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         1.54%        6.572s        12.02%       51.190s     511.900ms           100  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         1.11%        4.713s         6.66%       28.388s      94.627ms           300  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         1.02%        4.359s         1.02%        4.359s      21.795ms           200  
              enumerate(DataPipe)#NormalizeIterDataPipe         0.74%        3.158s         8.40%       35.780s     102.230ms           350  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         0.67%        2.847s         0.67%        2.854s      57.076ms            50  
enumerate(DataPipe)#SelectOverlappingTimeSliceIterDa...         0.36%        1.533s         1.83%        7.779s      77.793ms           100  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 425.979s

Double the size, to 64x64 (16x16 failed for some reason)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         enumerate(DataPipe)#CreateSunImageIterDataPipe        92.42%     1510.956s        92.42%     1510.958s       30.219s            50  
       enumerate(DataPipe)#ThreadPoolMapperIterDataPipe         2.98%       48.783s         5.23%       85.433s     427.163ms           200  
                 enumerate(DataPipe)#MapperIterDataPipe         1.33%       21.812s         1.78%       29.159s     145.796ms           200  
       enumerate(DataPipe)#OpenPVFromNetCDFIterDataPipe         1.23%       20.040s         1.23%       20.040s     200.403ms           100  
          enumerate(DataPipe)#CreatePVImageIterDataPipe         0.39%        6.454s         2.87%       46.984s     469.841ms           100  
enumerate(DataPipe)#SelectSpatialSlicePixelsIterData...         0.28%        4.576s         2.24%       36.649s      91.624ms           400  
              enumerate(DataPipe)#NormalizeIterDataPipe         0.24%        3.914s         2.21%       36.132s      80.293ms           450  
          enumerate(DataPipe)#OpenSatelliteIterDataPipe         0.22%        3.646s         0.22%        3.646s      18.232ms           200  
enumerate(DataPipe)#CreatePVMetadataImageIterDataPip...         0.18%        2.942s         0.18%        2.950s      58.997ms            50  
enumerate(DataPipe)#SelectOverlappingTimeSliceIterDa...         0.11%        1.842s         0.58%        9.409s      94.086ms           100  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1634.810s

Same as above with 16x16


"""

def test_irradiance_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = pseudo_irradiance_datapipe(filename, use_nwp=True, use_sat=True, use_hrv=True, use_sun=True, use_future=False, size=16)
    batch = next(iter(gsp_datapipe))
    batch = (torch.Tensor(batch[0]), torch.Tensor(batch[1]), torch.Tensor(batch[2]))
    x = np.nan_to_num(batch[0])
    assert np.isfinite(x).all()
    assert not np.isnan(batch[1]).any()
    assert np.isfinite(batch[2]).all()


def test_irradiance_datapipe_public_data():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test_public_data.yaml")
    gsp_datapipe = pseudo_irradiance_datapipe(filename, use_nwp=True, use_sat=True, use_hrv=True, use_sun=False, use_future=False, size=16)
    batch = next(iter(gsp_datapipe))
    batch = (torch.Tensor(batch[0]), torch.Tensor(batch[1]), torch.Tensor(batch[2]))
    x = np.nan_to_num(batch[0])
    assert np.isfinite(x).all()
    assert not np.isnan(batch[1]).any()
    assert np.isfinite(batch[2]).all()
