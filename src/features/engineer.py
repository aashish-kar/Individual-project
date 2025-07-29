"""
Feature engineering module for football player performance prediction.
"""

import pandas as pd
import numpy as np
import logging

class FeatureEngineer:
    """Handles feature engineering for the FPL dataset."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger(__name__)
    
    def create_all_features(self, df, target_column='FPL_points'):
        """
        Apply all feature engineering steps to the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Original dataset
        target_column : str
            Name of the target variable column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        self.logger.info("Creating engineered features")
        
        # Start with a copy to avoid modifying the original
        result = df.copy()
        
        # Apply feature engineering steps
        result = self.create_per_90_features(result)
        result = self.create_form_features(result, target_column)
        result = self.create_match_context_features(result)
        result = self.create_position_interaction_features(result)
        
        self.logger.info(f"Feature engineering complete: {result.shape[1]} total features")
        return result
    
    def create_per_90_features(self, df):
        """
        Create per-90 minute features for rate statistics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Player data with 'Min' column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added per-90 features
        """
        # Copy dataframe to avoid modifying the original
        result = df.copy()
        
        # Only process rows with positive minutes
        mask = result['Min'] > 0
        
        # List of rate statistics to convert to per-90
        rate_stats = ['xG', 'xA', 'Gls', 'Ast', 'Sh', 'SoT', 'Tkl', 'Int', 
                     'Blocks', 'SCA', 'GCA', 'PrgP', 'Carries', 'PrgC']
        
        # Only create features for columns that exist in the dataframe
        rate_stats = [col for col in rate_stats if col in result.columns]
        
        # Create per-90 features
        for stat in rate_stats:
            col_name = f"{stat}_per_90"
            result.loc[mask, col_name] = result.loc[mask, stat] * 90 / result.loc[mask, 'Min']
            
            # Cap extreme values that might result from very low minutes
            result[col_name] = result[col_name].clip(upper=result[col_name].quantile(0.99))
            
            # Fill NaN values (from rows with 0 minutes)
            result[col_name] = result[col_name].fillna(0)
            
            self.logger.debug(f"Created per-90 feature: {col_name}")
        
        self.logger.info(f"Created {len(rate_stats)} per-90 features")
        return result
    def create_form_features(self, df, target_column='FPL_points', window_sizes=[3, 5, 10]):
        """
        Create rolling form features based on past performance.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Player data with player identifier, season, and gameweek columns
        target_column : str
            Column to create form features from
        window_sizes : list
            List of window sizes for rolling averages
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added form features
        """
        # Copy dataframe to avoid modifying the original
        result = df.copy()
        
        # Check if required columns exist
        required_cols = ['Clean_name', 'Season', 'GW', target_column]
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns for form features: {missing_cols}")
            return result
        
        # Sort by player, season, and gameweek
        result = result.sort_values(['Clean_name', 'Season', 'GW'])
        
        # Create rolling averages for different window sizes
        for window in window_sizes:
            result[f'rolling_{target_column}_{window}'] = result.groupby(['Clean_name', 'Season'])[target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.logger.debug(f"Created {window}-match rolling average for {target_column}")
        
        # Create exponentially weighted moving average with more weight on recent matches
        result[f'exp_weighted_{target_column}'] = result.groupby(['Clean_name', 'Season'])[target_column].transform(
            lambda x: x.ewm(span=5).mean()
        )
        self.logger.debug(f"Created exponentially weighted moving average for {target_column}")
        
        # Create consistency metric (standard deviation over last 5 matches)
        result[f'consistency_{target_column}'] = result.groupby(['Clean_name', 'Season'])[target_column].transform(
            lambda x: x.rolling(window=5, min_periods=3).std()
        )
        result[f'consistency_{target_column}'] = result[f'consistency_{target_column}'].fillna(0)
        self.logger.debug(f"Created consistency metric for {target_column}")
        
        self.logger.info(f"Created {len(window_sizes) + 2} form features")
        return result
    
    def create_match_context_features(self, df):
        """
        Create features related to match context.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Player data with team and opponent information
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added match context features
        """
        # Copy dataframe to avoid modifying the original
        result = df.copy()
        
        # Create team rating difference if ratings are available
        if all(col in result.columns for col in ['Team_rating', 'Opp_rating']):
            result['rating_diff'] = result['Team_rating'] - result['Opp_rating']
            self.logger.debug("Created team rating difference feature")
        
        # Create match importance feature if end of season
        if 'GW' in result.columns:
            # Late season games (GW > 30) might have higher stakes
            result['is_late_season'] = (result['GW'] > 30).astype(int)
            self.logger.debug("Created late season indicator feature")
        
        # Create home advantage feature if not already present
        if 'Was_home' not in result.columns and 'Venue' in result.columns:
            result['Was_home'] = result['Venue'].apply(lambda x: 1 if x == 'Home' else 0)
            self.logger.debug("Created home advantage feature from Venue")
        
        # Calculate days rest if date information is available
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date'])
            # For each player and season, calculate days since last match
            result['days_rest'] = result.groupby(['Clean_name', 'Season'])['Date'].diff().dt.days
            # Fill first match of season with median rest days
            result['days_rest'] = result['days_rest'].fillna(result['days_rest'].median())
            self.logger.debug("Created days rest feature")
        
        self.logger.info("Created match context features")
        return result
    
    def create_position_interaction_features(self, df):
        """
        Create position-specific interaction features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Player data with position and performance columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added position interaction features
        """
        # Copy dataframe to avoid modifying the original
        result = df.copy()
        
        # Check if position column exists
        if 'FPL_pos' not in result.columns:
            self.logger.warning("Position column (FPL_pos) not found - skipping position interactions")
            return result
        
        # Create position indicators
        position_map = {
            'GKP': 0,
            'DEF': 1,
            'MID': 2,
            'FWD': 3
        }
        
        # Convert position to numeric if it's a string
        if result['FPL_pos'].dtype == 'object':
            result['pos_value'] = result['FPL_pos'].map(position_map)
        else:
            result['pos_value'] = result['FPL_pos']
        
        # Create position-based features
        if 'xG' in result.columns:
            # Attack expectation relative to position
            # Higher xG is more valuable for defenders than forwards
            result['xG_pos_value'] = result['xG'] * (4 - result['pos_value'])
            self.logger.debug("Created position-weighted xG feature")
        
        if 'Tkl' in result.columns and 'Int' in result.columns:
            # Defensive actions more valuable for attacking players
            result['defensive_contribution'] = (result['Tkl'] + result['Int']) * result['pos_value']
            self.logger.debug("Created position-weighted defensive contribution feature")
        
        # Create position-specific features
        # Goalkeepers
        gkp_mask = result['FPL_pos'] == 'GKP'
        if 'Player_CS' in result.columns and gkp_mask.sum() > 0:
            # Rolling clean sheet ratio for goalkeepers
            result.loc[gkp_mask, 'rolling_CS_ratio'] = result.loc[gkp_mask].groupby(['Clean_name', 'Season'])['Player_CS'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            self.logger.debug("Created goalkeeper-specific features")
        
        # Defenders
        def_mask = result['FPL_pos'] == 'DEF'
        if 'Player_CS' in result.columns and 'Team_CS' in result.columns and def_mask.sum() > 0:
            # Clean sheet probability for defenders based on team history
            result.loc[def_mask, 'clean_sheet_probability'] = result.loc[def_mask, 'Team_CS']
            self.logger.debug("Created defender-specific features")
        
        # Midfielders
        mid_mask = result['FPL_pos'] == 'MID'
        if all(col in result.columns for col in ['xA', 'Cmp', 'PrgP']) and mid_mask.sum() > 0:
            # Playmaking index for midfielders
            result.loc[mid_mask, 'playmaking_index'] = (
                result.loc[mid_mask, 'xA'] * 5 + 
                result.loc[mid_mask, 'PrgP'] * 0.2
            )
            self.logger.debug("Created midfielder-specific features")
        
        # Forwards
        fwd_mask = result['FPL_pos'] == 'FWD'
        if all(col in result.columns for col in ['xG', 'Sh', 'SoT']) and fwd_mask.sum() > 0:
            # Conversion quality for forwards
            with np.errstate(divide='ignore', invalid='ignore'):
                result.loc[fwd_mask, 'conversion_quality'] = np.where(
                    result.loc[fwd_mask, 'Sh'] > 0,
                    result.loc[fwd_mask, 'SoT'] / result.loc[fwd_mask, 'Sh'],
                    0
                )
            self.logger.debug("Created forward-specific features")
        
        self.logger.info("Created position interaction features")
        return result