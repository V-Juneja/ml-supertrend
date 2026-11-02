from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ml_supertrend import MLSupertrend,SupertrendConfig,BatchSample

class LearningTests(unittest.TestCase):
    def fire(self,buf,N=4,stride=2):
        model=MLSupertrend(SupertrendConfig(enable_tick_pressure=False))
        return model._batch_fire(buf,N,stride,2,.04,1,.02,.02,.01)
    def test_insufficient_completed_batch_cannot_update(self):
        buf=[BatchSample(1,0,1,1)]*3
        self.assertFalse(self.fire(buf)[0]);self.assertEqual(len(buf),3)
    def test_winning_batch_tightens_bands_and_consumes_only_stride(self):
        buf=[BatchSample(1,0,1,1) for _ in range(4)]
        result=self.fire(buf);self.assertTrue(result[0]);self.assertLess(result[1],0);self.assertLess(result[2],0);self.assertEqual(len(buf),2)
    def test_losing_batch_widens_bands(self):
        result=self.fire([BatchSample(-1,1,-1,1) for _ in range(4)])
        self.assertTrue(result[0]);self.assertGreater(result[1],0);self.assertGreater(result[2],0)
    def test_adverse_excursion_reduces_effective_batch_confidence(self):
        smooth=self.fire([BatchSample(1,0,1,1) for _ in range(4)])
        adverse=self.fire([BatchSample(1,3,1,1) for _ in range(4)])
        self.assertLess(abs(adverse[1]),abs(smooth[1]))
if __name__=='__main__':unittest.main()
