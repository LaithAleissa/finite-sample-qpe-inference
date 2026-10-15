"""Apply the common known-channel envelope to the prespecified material trial."""
from studies import *
from inference import difference_set
def main():
    rr=[json.loads(s) for s in (SRC/'known_noise/results/raw_counts.jsonl').read_text().splitlines() if json.loads(s)['stage']=='gap' and json.loads(s)['trial']==0]
    assert [r['case'] for r in rr]==['singlet','triplet_plus']
    sets=[]
    for r in rr:
        ex=Experiment(r['n'],r['time'],tuple(r['bounds']),r['assumed_uniform_noise']);obj=Inference(ex,r['batch_a'],r['batch_b'],.025)
        sets.append(known_invert(obj))
    gap=difference_set(sets[1]["intervals"],sets[0]["intervals"]);scale=113.30367910343477
    old=read(SRC/'material_model/results/propagation_summary.json')
    result=dict(old,known_noise_trial=0,gap_confidence_set_K=[[lo*scale,hi*scale] for lo,hi in gap],sets=sets,production_envelope='common Taylor/global envelope; terminal bin-width /10000',inherited_set_K=old['gap_confidence_set_K'])
    assert len(gap)==1 and gap[0][0]<=4<=gap[0][1]
    save(ROOT/'results/material_refined.json',result);print(result['gap_confidence_set_K'])
if __name__=='__main__':main()
