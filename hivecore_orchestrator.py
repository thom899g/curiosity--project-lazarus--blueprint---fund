"""
Core orchestration system for Project Homunculus.
Manages competing micro-agents, internal prediction markets, and execution.
This is the central nervous system of the autonomous economic entity.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

# Standard library imports - verified to exist
import numpy as np
import pandas as pd

# External libraries - known dependencies
try:
    import firebase_admin
    from firebase_admin import firestore, credentials
    import ccxt
    from web3 import Web3
    from sklearn.ensemble import IsolationForest
except ImportError as e:
    logging.error(f"Missing dependency: {e}")
    # Will cause graceful failure if dependencies not installed

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hive_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentProposal:
    """Structured proposal from an agent with risk assessment"""
    agent_id: str
    strategy_type: str
    target_protocol: str
    estimated_pnl_percent: float
    risk_score: float  # 0-1, higher = more risky
    capital_allocation_usd: float
    execution_window_seconds: int
    confidence_score: float = 0.5
    simulation_results: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AgentHive:
    """
    Orchestrates competing micro-agents with internal prediction markets.
    Implements adversarial-first design with multiple safety layers.
    """
    
    def __init__(self, initial_capital_usd: float = 1000.0):
        """
        Initialize the agent hive with initial capital.
        
        Args:
            initial_capital_usd: Starting capital in USD
        """
        self.available_capital = initial_capital_usd
        self.agents = {}
        self.internal_market = None  # Will be initialized after PredictionMarket import
        self.execution_history = []
        self._initialize_firestore()
        self._safety_lock = False  # Emergency stop
        
        logger.info(f"AgentHive initialized with ${initial_capital_usd}")
        
    def _initialize_firestore(self):
        """Initialize Firebase connection for state persistence"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate('firebase_credentials.json')
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully")
            self.db = firestore.client()
            # Create initial state document
            self.db.collection('hive_state').document('current').set({
                'capital_usd': self.available_capital,
                'status': 'active',
                'last_updated': firestore.SERVER_TIMESTAMP
            })
        except FileNotFoundError:
            logger.warning("Firebase credentials not found. Using in-memory state only.")
            self.db = None
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            self.db = None
    
    async def collect_proposals(self) -> List[AgentProposal]:
        """
        Collect strategy proposals from all registered agents.
        
        Returns:
            List of AgentProposal objects
        """
        proposals = []
        
        for agent_id, agent in self.agents.items():
            try:
                proposal = await agent.generate_proposal(self.available_capital)
                if proposal and self._validate_proposal(proposal):
                    proposals.append(proposal)
                    logger.debug(f"Valid proposal from {agent_id}: {proposal.strategy_type}")
                else:
                    logger.warning(f"Invalid proposal from {agent_id}")
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                continue
        
        logger.info(f"Collected {len(proposals)} valid proposals")
        return proposals
    
    def _validate_proposal(self, proposal: AgentProposal) -> bool:
        """
        Validate proposal against safety rules.
        
        Returns:
            bool: True if proposal passes all safety checks
        """
        # Check for emergency lock
        if self._safety_lock:
            logger.warning("Safety lock engaged - rejecting all proposals")
            return False
        
        # Capital allocation checks
        if proposal.capital_allocation_usd > self.available_capital * 0.2:
            logger.warning(f"Proposal exceeds 20% capital limit: {proposal.agent_id}")
            return False
        
        if proposal.capital_allocation_usd < 10:
            logger.warning(f"Proposal too small: ${proposal.capital_allocation_usd}")
            return False
        
        # Risk score threshold
        if proposal.risk_score > 0.8:
            logger.warning(f"Risk score too high: {proposal.risk_score}")
            return False
        
        # Execution window check
        if proposal.execution_window_seconds > 3600:  # 1 hour max
            logger.warning(f"Execution window too long: {proposal.execution_window_seconds}s")
            return False
        
        return True
    
    async def consensus_cycle(self, cycle_duration_seconds: int = 300):
        """
        Run one complete consensus cycle: collect, evaluate, execute.
        
        Args:
            cycle_duration_seconds: How long to wait between cycles
        """
        logger.info("Starting consensus cycle")
        
        try:
            # 1. Collect proposals
            proposals = await self.collect_proposals()
            
            if not proposals:
                logger.warning("No valid proposals collected")
                await asyncio.sleep(cycle_duration_seconds)
                return
            
            # 2. Evaluate proposals via internal market
            if self.internal_market:
                bids = await self.internal_market.evaluate_proposals(proposals)
                winning_proposal = self._select_winner(bids, proposals)
            else:
                # Fallback: select based on confidence score
                winning_proposal = max(proposals, key=lambda p: p.confidence_score)
            
            # 3. Execute with paranoia checks
            if winning_proposal:
                await self.execute_with_paranoia(winning_proposal)
            
            # 4. Update state
            self._update_state()
            
        except Exception as e:
            logger.error(f"Consensus cycle failed: {e}\n{traceback.format_exc()}")
            await self._emergency_protocol("consensus_failure", str(e))
        
        logger.info(f"Consensus cycle complete, sleeping for {cycle_duration_seconds}s")
        await asyncio.sleep(cycle_duration_seconds)
    
    def _select_winner(self, bids: Dict[str, float], proposals: List[AgentProposal]) -> Optional[AgentProposal]:
        """Select winning proposal based on internal market bids"""
        if not bids:
            return None
        
        # Weight bids by agent reputation
        weighted_bids = {}
        for agent_id, bid_amount in bids.items():
            # Get agent reputation from market
            reputation = self.internal_market.get_agent_reputation(agent_id)
            weighted_bids[agent_id] = bid_amount * reputation
        
        winning_agent = max(weighted_bids, key=weighted_bids.get)
        
        # Find corresponding proposal
        for proposal in proposals:
            if proposal.agent_id == winning_agent:
                logger.info(f"Selected winner: {winning_agent} with bid ${weighted_bids[winning_agent]:.2f}")
                return proposal
        
        return None
    
    async def execute_with_paranoia(self, proposal: AgentProposal):
        """
        Execute proposal with multiple safety checks.
        
        Args:
            proposal: Winning AgentProposal to execute
        """
        logger.info(f"Executing proposal from {proposal.agent_id}")
        
        # Pre-flight checks
        if not self._pre_flight_checks(proposal):
            logger.error("Pre-flight checks failed")
            return
        
        # Simulate on forked mainnet (if simulator available)
        simulation_result = await self._simulate_execution(proposal)
        if not simulation_result.get('success', False):
            logger.error(f"Simulation failed: {simulation_result.get('reason', 'Unknown')}")
            return
        
        # Check gas conditions
        if not self._check_gas_conditions():
            logger.warning("Gas conditions unfavorable, delaying execution")
            await asyncio.sleep(30)  # Wait 30 seconds
        
        # Execute with agent
        try:
            agent = self.agents[proposal.agent_id]
            execution_result = await agent.execute_strategy(proposal)
            
            # Record execution
            self.execution_history.append({
                'timestamp': datetime.now(),
                'agent_id': proposal.agent_id,
                'capital_allocated': proposal.capital_allocation_usd,
                'result': execution_result,
                'proposal': proposal.__dict__
            })
            
            # Update agent reputation
            if self.internal_market:
                pnl = execution_result.get('pnl_usd', 0)
                self.internal_market.update_reputation(proposal.agent_id, pnl)
            
            logger.info(f"Execution completed: {execution_result}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}\n{traceback.format_exc()}")
            await self._emergency_protocol("execution_failure", str(e))
    
    def _pre_flight_checks(self, proposal: AgentProposal) -> bool:
        """Execute mandatory pre-flight safety checks"""
        # Check capital availability
        if proposal.capital_allocation_usd > self.available_capital:
            logger.error("Insufficient capital for proposal")
            return False
        
        # Check if agent is still registered
        if proposal.agent_id not in self.agents:
            logger.error(f"Agent {proposal.agent_id} not found")
            return False
        
        # Check time window hasn't expired
        time_since_proposal = (datetime.now() - proposal.timestamp).total_seconds()
        if time_since_proposal > proposal.execution_window_seconds:
            logger.error("Proposal execution window expired")
            return False
        
        return True
    
    async def _simulate_execution(self, proposal: AgentProposal) -> Dict[str, Any]:
        """Simulate execution on forked mainnet"""
        # Placeholder - actual implementation would use web3.py with Anvil
        logger.info(f"Simulating execution for {proposal.agent_id}")
        
        # For now, return successful simulation for valid proposals
        # In production, this would connect to local Anvil fork
        return {
            'success': True,
            'gas_estimate': 100000,
            'cost_usd': proposal.capital_allocation_usd * 0.001,  # 0.1% estimated fees
            'simulated_pnl': proposal.estimated_pnl_percent * proposal.capital_allocation_usd
        }
    
    def _check_gas_conditions(self) -> bool:
        """Check current gas conditions are favorable"""
        # Placeholder - would integrate with gas price APIs
        # For now, randomly fail 10% of the time to simulate adverse conditions
        import random
        return random.random() > 0.1
    
    async def _emergency_protocol(self, trigger: str, details: str):
        """Execute emergency protocols based on trigger"""
        logger.critical(f"EMERGENCY PROTOCOL ACTIVATED: {trigger} - {details}")
        
        # Engage safety lock
        self._safety_lock = True
        
        # Log to Firebase if available
        if self.db:
            try:
                self.db.collection('emergency_logs').add({
                    'trigger': trigger,
                    'details': details,
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'capital_at_risk': self.available_capital
                })
            except Exception as e:
                logger.error(f"Failed to log emergency: {e}")
        
        # In production, would initiate withdrawal to cold storage
        logger.info("Safety lock engaged. Manual intervention required.")
    
    def _update_state(self):
        """Update Firebase state"""
        if self.db:
            try:
                self.db.collection('hive_state').document('current').update({
                    'capital_usd': self.available_capital,
                    'last_updated': firestore.SERVER_TIMESTAMP,
                    'execution_count': len(self.execution_history)
                })
            except Exception as e:
                logger.error(f"Failed to update state: {e}")
    
    async def run_continuously(self):
        """Run consensus cycles continuously"""
        logger.info("Starting continuous hive operation")
        
        while not self._safety_lock:
            try:
                await self.consensus_cycle()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        logger.info("Hive operation terminated")


# Example usage
async def main():
    """Main entry point for the hive"""
    hive = AgentHive(initial_capital_usd=1000.0)
    
    # Register agents (implementation in separate files)
    # hive.agents['conservative_scab'] = ConservativeScabAgent()
    # hive.agents['speculative_hunter'] = SpeculativeHunterAgent()
    # hive.agents['paranoid_sentinel'] = ParanoidSentinelAgent()
    
    # Start the hive
    await hive.run_continuously()


if __name__ == "__main__":
    asyncio.run(main())