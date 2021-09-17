import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

st.title('Bayesian A/B test calculator')

###INPUT prior belief
# prior = st.slider('Prior Belief', min_value=0, max_value=100, value=50)

###INPUT variation A,B users and conversions
col1, col2 = st.columns(2)
col1.header('Variation A:')
orig_users = col1.number_input('Visitors', step=1, value=1000, key='orig_users')
orig_converted = col1.slider('Converted', min_value=0, max_value=orig_users, value=100, key='orig_converted')
#col1.number_input('Converted', step=1, value=100, key='orig_converted')
col2.header('Variation B:')
variation_users = col2.number_input('Visitors', step=1, value=1005, key='variation_users')
variation_converted = col2.slider('Converted', min_value=0, key='variation_converted', max_value=variation_users, value=101)
# col2.number_input('Converted', step=1, value=101, key='variation_converted')

###CALCULATION of Probabilities for the PDF function
orig_non_converted = orig_users - orig_converted
orig_probs = np.linspace(
    beta.ppf(0.01, orig_converted, orig_non_converted), 
    beta.ppf(0.99, orig_converted, orig_non_converted),
    100)

variation_non_converted = variation_users - variation_converted
variation_probs = np.linspace(
    beta.ppf(0.01, variation_converted, variation_non_converted), 
    beta.ppf(0.99, variation_converted, variation_non_converted),
    100)

###CALCULATION PDP and CDF
origin_pdf = beta.pdf(orig_probs, orig_converted, orig_non_converted)
origin_cdf = beta.cdf(orig_probs, orig_converted, orig_non_converted)

variation_pdf = beta.pdf(variation_probs, variation_converted, variation_non_converted)
variation_cdf = beta.cdf(variation_probs, variation_converted, variation_non_converted)

###draw PDF and CDF
charts_col1, charts_col2 = st.columns(2)
pdf_fig = plt.figure(figsize=(8,8))
pdf_ax = pdf_fig.subplots()
pdf_ax.plot(orig_probs, origin_pdf, label='variation A')
pdf_ax.plot(variation_probs, variation_pdf, label='variation B')
pdf_ax.legend()
pdf_ax.set_xlabel('conversions')
pdf_ax.set_ylabel('probability')
charts_col1.header('Probability dencity function')
charts_col1.pyplot(pdf_fig)

cdf_fig = plt.figure(figsize=(8,8))
cdf_ax = cdf_fig.subplots()
cdf_ax.plot(orig_probs, origin_cdf, label='variation A')
cdf_ax.plot(variation_probs, variation_cdf, label='variation B')
cdf_ax.legend()
cdf_ax.set_xlabel('conversions')
cdf_ax.set_ylabel('probability')
charts_col2.header('Cumulative distribution function')
charts_col2.pyplot(cdf_fig)

###Probability to be best
## Monte Carlo integration (importance Sampling)
trials = 10000
orig_samples = np.random.beta(orig_converted, orig_non_converted, trials)
variant_samples = np.random.beta(variation_converted,variation_non_converted, trials)
b_better_trials = np.sum(variant_samples >= orig_samples)
p = b_better_trials / trials
st.title('Probability B > A: %s%%' % int(p*100))
st.title('Probability A > B: %s%%' % int((1-p)*100))