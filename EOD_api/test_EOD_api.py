import os
import re
import datetime
import unittest
from io import StringIO
from unittest.mock import patch

import pandas as pd

import EOD_api as eod

# token = os.environ['EOD_TOKEN'] 
token= '12ab3cd5efgh45.12345678'
token='5fb8d1beabe962.82448453'

def date_parser( string ):
    date_pattern = re.compile('([0-9]{4}-[0-9]{2}-[0-9]{2})[ ]', re.VERBOSE)
    return date_pattern.sub(r'\1T', string)

class TestGetEod(unittest.TestCase):
    # @classmethod
    # def setUp(self):
    #     pass
    # def tearDown(self):
    #     pass

    def test_idempotent__addtickers( self ):
        d1 = eod.ohlcv_intraday( ['AAPL.US'], token, '2020-10-13', '2020-10-17',intraday_frec='5m').add_tickers( ['MSFT.US'])
        d2 = eod.ohlcv_intraday( ['AAPL.US'], token, '2020-10-13', '2020-10-17',intraday_frec='5m').add_tickers( ['MSFT.US']).add_tickers( ['MSFT.US'])
        self.assertEqual( d1, d2)

    def test_idempotent_truncate_dates( self ):
        d1 = eod.fundamental( ['AAPL.US'], token, '2020-10-13', '2020-10-17').truncate_dates('2020-10-14', '2020-10-16')
        d2 = eod.fundamental( ['AAPL.US'], token, '2020-10-13', '2020-10-17').truncate_dates('2020-10-14', '2020-10-16').truncate_dates('2020-10-14', '2020-10-16')
        self.assertEqual( d1, d2)
    
    def test_idempotent_remove_tickers( self ):
        d1 = eod.fundamental( ['AAPL.US','MSFT.US'], token, '2020-10-13', '2020-10-17').remove_tickers( ['MSFT.US'] )
        d2 = eod.fundamental( ['AAPL.US','MSFT.US'], token, '2020-10-13', '2020-10-17').remove_tickers( ['MSFT.US'] ).remove_tickers( ['MSFT.US'] )
        self.assertEqual( d1, d2)

    def test_add_remove( self ): 
        d1 = eod.ohlcv_intraday( ['AAPL.US'], token, '2020-10-13', '2020-10-17','1m')
        d2 = eod.ohlcv_intraday( ['AAPL.US'], token, '2020-10-13', '2020-10-17','1m').add_tickers( ['MSFT.US'] ).remove_tickers( ['MSFT.US'] )
        self.assertEqual( d1, d2)

    def test_remove_all_tickers( self ):
        with self.assertRaises(Exception) :
            eod.ohlcv(['AAPL.US'], token, '2020-10-13', '2020-10-17').remove_tickers(['AAPL.US']).retrieve_data()

    def test_misspelled_input( self ):
        with self.assertRaises(Exception) :
            eod.ohlcv_intraday(['AAPL.US'], token, '2020-10-13', '2020-10-17', intraday_frec='Daoly')

    def test_ohlcv_data_format_hasnt_changed(self): # Cambiar de antes de formatting a después de formatting
        expected_aapl = pd.read_csv( StringIO(  """
            Date          Open     High     Low   Close  Adjusted_close       Volume
            2020-10-13  125.27  125.390  119.65  121.10        120.7110  262330500.0
            2020-10-14  121.00  123.030  119.62  121.19        120.8008  151062297.0
            2020-10-15  118.72  121.200  118.15  120.71        120.3223  112559203.0
            2020-10-16  121.28  121.548  118.81  119.02        118.6377  115393797.0
                275     NaN      NaN     NaN     NaN             NaN          NaN
            """ ), sep="\\s+" )

        url = 'https://eodhistoricaldata.com/api/eod/AAPL.US?api_token={}&from=2020-10-13&to=2020-10-17&period=d'.format(token)
        actual = pd.read_csv(url, usecols = ['Date','Volume','Open','Close','High','Low','Adjusted_close'] )
        with patch.object(pd,'read_csv') as mock_read:
            mock_read.autospec = True
            mock_read.return_value = expected_aapl
            expected = pd.read_csv(url, usecols = ['Date','Volume','Open','Close','High','Low','Adjusted_close'] )
        pd.testing.assert_frame_equal(actual, expected, rtol = 5e-3)

    def test_index_formatting( self):
        expected_aapl = pd.read_csv( StringIO(  """
            Date          Open     High     Low   Close  Adjusted_close       Volume
            2020-10-13  125.27  125.390  119.65  121.10        120.7110  262330500.0
            2020-10-14  121.00  123.030  119.62  121.19        120.8008  151062297.0
            2020-10-15  118.72  121.200  118.15  120.71        120.3223  112559203.0
            2020-10-16  121.28  121.548  118.81  119.02        118.6377  115393797.0
                275     NaN      NaN     NaN     NaN             NaN          NaN
            """ ), sep="\\s+" )
        expected_aapl_formatted = pd.read_csv( StringIO( date_parser( """
        Stock   Date                         Open     High     Low   Close  Adjusted_close       Volume                                                              
        AAPL.US 2020-10-13 00:00:00+00:00  125.27  125.390  119.65  121.10        120.7110  262330500.0
        AAPL.US 2020-10-14 00:00:00+00:00  121.00  123.030  119.62  121.19        120.8008  151062297.0
        AAPL.US 2020-10-15 00:00:00+00:00  118.72  121.200  118.15  120.71        120.3223  112559203.0
        AAPL.US 2020-10-16 00:00:00+00:00  121.28  121.548  118.81  119.02        118.6377  115393797.0
        """ ) ), sep="\\s+", index_col=[0,1], converters = {'Date' : lambda col: datetime.datetime.fromisoformat( col ) } ) 

        with patch.object(pd,'read_csv') as mock_read:
            mock_read.autospec = True
            mock_read.return_value = expected_aapl
            formatted_mock = eod.ohlcv(['AAPL.US'], token, '2020-10-13', '2020-10-17').retrieve_data()
        pd.testing.assert_frame_equal(formatted_mock, expected_aapl_formatted, rtol = 5e-3)

# TODO? Write more tests:
# Check that the data is concated/merged/joined properly, particularly when the indexes come with Nans
# Check except clauses
# Check duplicate df values 
# Assert errors with wrong args
# etc

# expected_ohlcv_concatted = pd.read_csv( StringIO( date_parser( """
# Stock     Date                       Gmtoffset             Datetime        Open        High         Low       Close     Volume   Returns                                                                                                       
# BP.LSE    2020-10-13 00:00:00+00:00        NaN                  NaN         NaN         NaN         NaN         NaN        NaN       NaN
# BP.LSE    2020-10-14 00:00:00+00:00        0.0  2020-10-13 15:25:00  213.649993  214.000000  213.550003  213.856994  1210380.0 -0.001601
# BP.LSE    2020-10-15 00:00:00+00:00        0.0  2020-10-14 15:25:00  213.000000  213.149993  212.600006  212.649993  1182246.0  0.019660
# BP.LSE    2020-10-16 00:00:00+00:00        0.0  2020-10-15 15:25:00  207.149993  207.199996  206.500000  206.850006  1626720.0 -0.013826
# AAPL.US   2020-10-13 00:00:00+00:00        NaN                  NaN         NaN         NaN         NaN         NaN        NaN       NaN
# AAPL.US   2020-10-14 00:00:00+00:00        0.0  2020-10-13 19:55:00  121.139999  121.279998  121.029998  121.050003  4585723.0  0.003648
# AAPL.US   2020-10-15 00:00:00+00:00        0.0  2020-10-14 19:55:00  121.580001  121.709999  121.139999  121.180000  3420583.0  0.015419
# AAPL.US   2020-10-16 00:00:00+00:00        0.0  2020-10-15 19:55:00  120.790000  120.849998  120.580001  120.699996  3436603.0 -0.003550
# MSFT.US   2020-10-13 00:00:00+00:00        NaN                  NaN         NaN         NaN         NaN         NaN        NaN       NaN
# MSFT.US   2020-10-14 00:00:00+00:00        0.0  2020-10-13 19:55:00  223.320007  223.389999  222.750000  222.830001  1457493.0  0.000651
# MSFT.US   2020-10-15 00:00:00+00:00        0.0  2020-10-14 19:55:00  221.199996  221.414993  220.600006  220.759994  1122912.0  0.012377
# MSFT.US   2020-10-16 00:00:00+00:00        0.0  2020-10-15 19:55:00  219.639999  219.880004  219.490005  219.660003  1201342.0 -0.003900
# """ ) ), sep="\\s+", index_col=[0,1,2], converters = {'Date' : lambda col: datetime.datetime.fromisoformat( col ) \
# , 'Datetime' : lambda col: pd.to_datetime(col, format='%Y-%m-%dT%H:%M:%S', utc=True) }    )

if __name__ == "__main__":
     unittest.main()





