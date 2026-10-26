"""Additional facts from the original complete MBO publications.

T is counted once. F only attributes a resting decrement, never a second trade.
An unlinked decrement is a withdrawal proxy, not proof of cancellation intent.
"""
from collections import defaultdict, deque
import numpy as np
from spectrum.core.event_facts import decode_event

NS=10**9
FIELDS=('add','remove','linked_fill','passive_fill','aggressor_fill','ambiguous_remove',
        'unlinked_remove','refill','refill_delay_qseconds','removed_age_qseconds','removed_age_known_qty',
        'near_add','near_remove','near_fill','near_withdrawal','near_ambiguous','near_refill')


class InteractionCollector:
    def __init__(self, emit_levels, emit_trades):
        self.emit_levels=emit_levels;self.emit_trades=emit_trades
        self.bucket=None;self.rows={};self.recent={};self.birth={};self.shadow={}
        self.stats=dict(events=0,supported_events=0,stock_checks=0,max_stock_residual=0,
                        trade_qty=0,unresolved_fill=0,ambiguous_remove=0,linked_fill=0,
                        unlinked_remove=0,refill=0,removed_qty=0,removed_age_known_qty=0,
                        legacy_attribution_changed_qty=0,partial_end_level_rows=0)

    def flush(self, available_ns):
        if self.rows:
            rows=[]
            for (side,price),v in self.rows.items():
                rows.append(dict(t_ns=self.bucket,available_ns=int(available_ns),side=side,
                    price_nano=price,**dict(zip(FIELDS,v))))
            self.emit_levels(rows)
        self.rows={}

    def feed(self,event,engine):
        t=int(event.available_at_ns);bucket=(t//NS+1)*NS
        if self.bucket is not None and bucket!=self.bucket:self.flush(t)
        self.bucket=bucket;self.stats['events']+=1
        facts=decode_event(event)
        for tr in facts.trades:
            self.emit_trades(dict(t_ns=t,available_ns=t,ordinal=tr.ordinal,price_nano=tr.price_nano,
                quantity=tr.quantity,side=1 if tr.side=='B' else -1 if tr.side=='A' else 0,
                supported=bool(facts.supported),epoch=int(event.book.continuity_epoch)))
            self.stats['trade_qty']+=tr.quantity
        if not facts.supported:
            self.recent.clear();self.birth.clear()
            self.shadow={(s,int(p)):int(l.size) for s,levels in ((1,engine.bids),(-1,engine.asks)) for p,l in levels.items()}
            return
        self.stats['supported_events']+=1
        self.stats['unresolved_fill']+=facts.unresolved_fill
        self.stats['legacy_attribution_changed_qty']+=facts.legacy_changed_qty
        changed=set();fills={}
        for m in facts.mutations:
            if not m.accepted:continue
            side=1 if m.side=='B' else -1;key=(side,int(m.price_nano));changed.add(key)
            v=self.rows.setdefault(key,np.zeros(len(FIELDS),float))
            best=event.book_before.best_bid_nano if side==1 else event.book_before.best_ask_nano
            near=event.book_before.valid and 0<=side*(best-m.price_nano)<=11*100_000_000
            self.shadow[key]=self.shadow.get(key,0)+m.quantity
            if m.quantity<0:
                q=-m.quantity;withdraw=q-m.linked_fill-m.ambiguous_remove
                assert withdraw>=0
                v[1]+=q;v[2]+=m.linked_fill;v[3]+=m.passive_fill;v[4]+=m.aggressor_fill
                v[5]+=m.ambiguous_remove;v[6]+=withdraw
                self.stats['removed_qty']+=q;self.stats['linked_fill']+=m.linked_fill
                self.stats['ambiguous_remove']+=m.ambiguous_remove;self.stats['unlinked_remove']+=withdraw
                birth=self.birth.get((m.order_id,key))
                if birth is not None:
                    v[9]+=q*(t-birth)/NS;v[10]+=q;self.stats['removed_age_known_qty']+=q
                if near:v[12]+=q;v[13]+=m.linked_fill;v[14]+=withdraw;v[15]+=m.ambiguous_remove
                if m.linked_fill:fills[key]=fills.get(key,0)+m.linked_fill
            else:
                q=m.quantity;v[0]+=q
                if near:v[11]+=q
                # First observed addition is not a known birth if it amends a
                # pre-existing snapshot order. Only raw A/reprice below sets it.
                budgets=self.recent.get(key,deque())
                while budgets and t-budgets[0][0]>2*NS:budgets.popleft()
                remaining=q
                while budgets and remaining>0 and budgets[0][0]<t:
                    stamp,amount=budgets.popleft();r=min(remaining,amount)
                    v[7]+=r;v[8]+=r*(t-stamp)/NS;self.stats['refill']+=r
                    if near:v[16]+=r
                    remaining-=r
                    if amount>r:budgets.appendleft((stamp,amount-r))
                self.recent[key]=budgets
        for key,q in fills.items():
            self.recent.setdefault(key,deque()).append((t,q))
        for d in event.deltas:
            if not d.accepted or d.action not in ('A','C','M'):continue
            prior=(1 if d.prior_side=='B' else -1,int(d.prior_price_nano))
            result=(1 if d.resulting_side=='B' else -1,int(d.resulting_price_nano))
            if d.prior_size and (not d.resulting_size or prior!=result):self.birth.pop((d.order_id,prior),None)
            if d.resulting_size and (not d.prior_size or prior!=result):self.birth[(d.order_id,result)]=t
        for side,price in changed:
            actual=(engine.bids if side==1 else engine.asks).get(price)
            actual=0 if actual is None else actual.size
            residual=abs(actual-self.shadow[(side,price)])
            self.stats['stock_checks']+=1
            self.stats['max_stock_residual']=max(self.stats['max_stock_residual'],residual)
            if residual:raise ValueError('Stock reconciliation failed')
        if self.stats['events']%100000==0:
            self.recent={k:v for k,v in self.recent.items() if v and t-v[-1][0]<=2*NS}

    def finish(self):
        # No later publication proves completion of the last second.
        self.stats['partial_end_level_rows']=len(self.rows)
        self.rows={}
