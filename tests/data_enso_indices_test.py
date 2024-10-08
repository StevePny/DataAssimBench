"""Tests for ENSOIndices class (dabench.data.enso_indices)"""

from dabench.data import ENSOIndices
import pytest


def test_initialization():
    """Tests initialization of enso idx data"""
    file_dict = {'wnd': ['zwnd200'],
                 'slp': ['darwin']}
    var_types = {'wnd': ['ori'],
                 'slp': ['ori', 'std']}

    ds = ENSOIndices(file_dict, var_types).load()

    assert ds.system_dim == 3
    assert ds.dab.flatten().shape[1] == ds.system_dim


def test_values_wnd_slp():
    """Tests assorted values"""
    file_dict = {'wnd': ['zwnd200'],
                 'slp': ['darwin']}
    var_types = {'wnd': ['ori'],
                 'slp': ['ori']}

    ds = ENSOIndices(file_dict, var_types).load().dab.flatten()

    assert ds.data[0, 0] == 15.5
    assert ds.data[13, 0] == 7.9
    assert ds.data[0, 1] == 6.0
    assert ds.data[50, 1] == 10.1


def test_values_eqsoi():
    """Tests assorted values for eqsoi"""

    file_dict = {'eqsoi': ['rindo_slpa.for', 'reqsoi.3m.for']}
    var_types = {'eqsoi': ['std']}

    ds = ENSOIndices(file_dict, var_types).load().dab.flatten()

    assert ds.shape[1] == 2
    assert ds.data[0, 0] == -0.5
    assert ds.data[13, 0] == -0.9
    assert ds.data[0, 1] == -1.0
    assert ds.data[60, 1] == 0.1


def test_values_sst_rsst():
    """Tests values for sst and rsst"""
    file_dict = {'rsst': ['sstoi.atl.indices'],
                 'sst': ['ersst5.nino.mth.91-20.ascii']}
    var_types = {'rsst': ['tr_ano', 'sa'],
                 'sst': ['nino12', 'nino34_ano']}

    ds = ENSOIndices(file_dict, var_types).load().dab.flatten()

    assert ds.shape[1] == 4
    assert ds.data[0, 0] == -0.20
    assert ds.data[0, 1] == 25.26
    assert ds.data[0, 2] == 24.30
    assert ds.data[0, 3] == 0.13


def test_shape_all():
    """Tests shape of values when you get EVERYTHING"""
    file_dict_full = {'wnd': ['zwnd200', 'wpac850', 'cpac850', 'epac850',
                              'qbo.u30.index', 'qbo.u50.index'],
                      'slp': ['darwin', 'tahiti'],
                      'soi': ['soi'],
                      'soi3m': ['soi.3m.txt'],
                      'eqsoi': ['rindo_slpa.for', 'repac_slpa.for',
                                'reqsoi.for', 'reqsoi.3m.for'],
                      'sst': ['sstoi.indices',
                              'ersst5.nino.mth.91-20.ascii'],
                      'desst': ['detrend.nino34.ascii.txt'],
                      'rsst': ['sstoi.atl.indices'],
                      'olr': ['olr'],
                      'cpolr': ['cpolr.mth.91-20.ascii']
                      }
    var_types_full = {'wnd': ['ori', 'ano', 'std'],
                      'slp': ['ori', 'ano', 'std'],
                      'soi': ['ano', 'std'],
                      'soi3m': ['ori'],
                      'eqsoi': ['std'],
                      'sst': ['nino12', 'nino12_ano', 'nino3', 'nino3_ano',
                              'nino4', 'nino4_ano', 'nino34',
                              'nino34_ano'],
                      'desst': ['ori', 'adj', 'ano'],
                      'rsst': ['na', 'na_ano', 'sa', 'sa_ano', 'tr',
                               'tr_ano'],
                      'olr': ['ori', 'ano', 'std'],
                      'cpolr': ['ano']}

    ds = ENSOIndices(file_dict_full, var_types_full).load()

    assert ds.system_dim == 60
    assert ds.dab.flatten().shape[1] == 60



