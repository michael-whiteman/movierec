from __future__ import annotations
import json
import pathlib
import re

import numpy as np
import pandas as pd
import streamlit as st

from movierec.infer.ann import build_ip_index, query_index
from movierec.infer.personalize import apply_mmr, build_personal_vec, explain_recommendation, rerank_candidates


ART_ROOT = pathlib.Path('artifacts')
SPLITS_DIR = ART_ROOT / 'splits'
PROFILES_DIR = pathlib.Path('profiles')
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_profile():
    """Ensure user profile exists."""
    st.session_state.setdefault('profile', {
        'likes': [],
        'dislikes': [],
        'hidden': [],
        'watchlist': [],
        'genres': {},
        'year_pref': {'mu': 2000, 'sigma': 20},
        'weights': {'base': 1.0, 'sim': 0.7, 'genre': 0.2, 'year': 0.2, 'novel': 0.1, 'watch': 0.15, 'recency': 0.0},
        'ui': {'recency_half_life': 8, 'diversity_lambda': 0.85, 'watchlist_exclude': False}
    })
    return st.session_state['profile']


def render_genre_preferences(items: pd.DataFrame):
    prof = ensure_profile()
    with st.sidebar.expander('Genre Preferences', expanded=False):
        # Two-column label/slider layout to avoid text wrapping
        all_genres = sorted({g.strip() for s in items['genres'].dropna().tolist() for g in s.split('|')})
        max_show = 24
        n_show = max(1, min(max_show, len(all_genres)))
        genre_weights = {}

        for g in all_genres[:n_show]:
            lcol, scol = st.columns([2, 3])  # widen label column
            with lcol:
                st.markdown(f'**{g}**')
            with scol:
                genre_weights[g] = st.slider(
                    f'weight for {g}', 0.0, 1.0, float(prof.get('genres', {}).get(g, 0.0)), 0.1,
                    key=f'gen_{g}', label_visibility='collapsed'
                )
        prof['genres'] = {g: float(w) for g, w in genre_weights.items() if w > 0.0}

    # Push back to profile
    st.session_state['profile'] = prof


def render_personal_knobs():
    prof = ensure_profile()
    with st.sidebar.expander('Personalization Controls', expanded=True):
        prof['weights']['novel'] = st.slider(
            'Novelty boost', 0.0, 0.6, float(prof['weights'].get('novel', 0.1)), 0.05,
            help='reward less-popular items',
        )
        prof['weights']['recency'] = st.slider(
            'Recency boost', 0.0, 0.6, float(prof['weights'].get('recency', 0.0)), 0.05,
            help='prefer recent releases',
        )
        prof['ui']['recency_half_life'] = st.slider(
            'Recency half-life (Years)', 2, 30, int(prof['ui'].get('recency_half_life', 8)), 1,
            help='age that halves recency contribution',
        )
        prof['ui']['diversity_lambda'] = st.slider(
            'Diversity (MMR λ)', 0.5, 1.0, float(prof['ui'].get('diversity_lambda', 0.85)), 0.01,
            help='trade-off relevance vs diversity (1.0 = no diversification)',
        )
    st.session_state['profile'] = prof


def render_watchlist_sidebar(items_df: pd.DataFrame, key_prefix: str = 'wl'):
    """Sidebar manager shown in *both* tabs."""
    prof = ensure_profile()

    # Persist UI options
    with st.sidebar.expander('⭐ Watchlist', expanded=False):
        wl = prof.get('watchlist', [])
        if wl:
            for wid in list(map(int, wl)):
                title = re.sub(r'\s*\(\d{4}\)\s*$', '', str(items_df.loc[wid, 'title']))
                colA, colB = st.columns([4, 1])
                colA.write(title)
                colB.button('✖', key=f'{key_prefix}_rm_{wid}', on_click=watchlist_toggle, args=(wid,))
        else:
            st.caption('Empty. Add with the ⭐ button.')

    # Controls that persist to profile
    prof.setdefault('weights', prof.get('weights', {}))
    prof.setdefault('ui', prof.get('ui', {}))
    prof['ui']['watchlist_exclude'] = st.sidebar.checkbox(
        'Exclude watchlist',
        value=bool(prof['ui'].get('watchlist_exclude', False)),
        key=f'{key_prefix}_exclude'
    )

    # Write slider weights into profile weights (i.e., saved with the profile)
    prof['weights']['watch'] = st.sidebar.slider(
        'Watchlist boost', 0.0, 0.5, float(prof['weights'].get('watch', 0.15)), 0.05,
        key=f'{key_prefix}_boost', help='nudge watchlisted items up during re-ranking'
    )

    # Push back into session
    st.session_state['profile'] = prof


def render_year_preferences(items: pd.DataFrame):
    prof = ensure_profile()
    with st.sidebar.expander('Year Preferences', expanded=False):
        # Year priors
        default_mu = int(prof.get('year_pref', {}).get('mu', 2000))
        default_sigma = int(prof.get('year_pref', {}).get('sigma', 20))
        year_mu = st.slider('Preferred Year', 1910, 2025, default_mu, key='year_mu')
        year_sigma = st.slider('Year Spread', 5, 40, default_sigma, key='year_sigma')
        prof['year_pref'] = {'mu': int(year_mu), 'sigma': int(year_sigma)}

        # Decades to refine prior
        decades = sorted({decade_of(y) for y in items['year'].tolist()})
        pref_decades = st.multiselect('Decades You Like', decades, key='setup_decades')
        if pref_decades:
            centers = [int(d[:-1]) + 5 for d in pref_decades if d.endswith('s')]
            if centers:
                prof['year_pref'] = {
                    'mu': int(np.mean(centers)),
                    'sigma': min(prof['year_pref']['sigma'], 12)
                }
    
    # Push back to profile
    st.session_state['profile'] = prof


def watchlist_toggle(item_id: int, profile_key: str = 'profile'):
    """Add/remove an item from the watchlist in session state."""
    prof = st.session_state['profile']
    iid = int(item_id)
    wl = prof.setdefault('watchlist', [])
    if iid in wl:
        wl.remove(iid)
        prof['watchlist'] = sorted(wl)
    else:
        # Use the commit helper so other sets stay coherent
        commit_feedback_to_profile(prof, new_watchlist=[iid])
    st.session_state[profile_key] = prof


# ---------- General Helpers ----------
def apply_feedback(action: str, item_id: int, profile_key: str = 'profile'):
    prof = st.session_state['profile']
    iid = int(item_id)
    if action == 'like':
        commit_feedback_to_profile(prof, [iid], [], [])
    elif action == 'dislike':
        commit_feedback_to_profile(prof, [], [iid], [])
    elif action == 'hide':
        commit_feedback_to_profile(prof, [], [], [iid])
    st.session_state[profile_key] = prof


def commit_feedback_to_profile(profile: dict, new_likes=None, new_dislikes=None, new_hidden=None, new_watchlist=None):
    """Merge feedback into the profile dict (likes override others)."""
    new_likes = [int(x) for x in (new_likes or [])]
    new_dislikes = [int(x) for x in (new_dislikes or [])]
    new_hidden = [int(x) for x in (new_hidden or [])]
    new_watchlist = [int(x) for x in (new_watchlist or [])]

    like_set = set(map(int, profile.get('likes', [])))
    dislike_set = set(map(int, profile.get('dislikes', [])))
    hidden_set = set(map(int, profile.get('hidden', [])))
    watchlist_set = set(map(int, profile.get('watchlist', [])))

    # Likes
    like_set |= set(new_likes)

    # Remove from other buckets if newly liked
    if new_likes:
        dislike_set -= set(new_likes)
        hidden_set -= set(new_likes)

    # Dislikes / Hidden
    dislike_set |= (set(new_dislikes) - like_set)
    hidden_set |= (set(new_hidden) - like_set)

    # Watchlist
    if new_watchlist:
        watchlist_set |= set(new_watchlist)
        hidden_set -= watchlist_set  # If something is on watchlist, un-hide it (feels more natural)

    profile['likes'] = sorted(like_set)
    profile['dislikes']  = sorted(dislike_set)
    profile['hidden'] = sorted(hidden_set)
    profile['watchlist'] = sorted(watchlist_set)


def decade_of(year: float | int | None) -> str:
    if pd.isna(year):
        return 'Unknown'
    y = int(year)
    return f'{(y//10)*10}s'


def ensure_normalized_item_emb(I: np.ndarray) -> np.ndarray:
    I = I.astype('float32', copy=False)
    I = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-12)
    return I


def load_artifacts(dataset: str, model_name: str) -> Tuple[np.ndarray | None, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    dataset: 'ml-1m' or 'ml-32m'
    model_name: 'bpr_mf' or 'lightgcn'
    returns: U, I, items_df, train_df, val_df
    """
    root = ART_ROOT / dataset
    
    # Items & splits
    items = pd.read_parquet(root / 'items.parquet')
    if 'i' in items.columns:
        items = items.set_index('i').sort_index()
    
    # int index
    try:
        items.index = items.index.astype(int)
    except Exception:
        pass

    splits = root / 'splits'
    train = pd.read_parquet(splits / 'train.parquet')
    val   = pd.read_parquet(splits / 'val.parquet')

    # Embeddings
    model_dir = root / model_name
    U = np.load(model_dir / 'user.npy') if (model_dir / 'user.npy').exists() else None
    I = np.load(model_dir / 'item.npy')

    # Attach popularity for novelty
    pop = train['i'].value_counts()
    items['__pop__'] = pop.reindex(items.index).fillna(0).astype('int64')

    # Fill any missing ids up to I.shape[0]
    n_items = I.shape[0]
    present = set(items.index.astype(int).tolist())
    missing = sorted(set(range(n_items)) - present)
    if missing:
        filler = pd.DataFrame(
            {
                'title': [f'Item {mid} ({mid})' for mid in missing],
                'genres': [''] * len(missing),
                'year': [np.nan] * len(missing),
                '__pop__': [0] * len(missing),
            },
            index=pd.Index(missing, name=items.index.name or 'i'),
        )
        items = pd.concat([items, filler], axis=0).sort_index()

    return U, I, items, train, val

# ---------- app ----------

def main():
    st.set_page_config(page_title='Movie Recs', layout='wide', initial_sidebar_state='expanded')
    st.title("🎬 Recommend ⭐Me⭐ a Movie")

    # Persistent user id
    st.session_state.setdefault('user_id', 'demo')

    # Sidebar tabs: profile / model / watchlist
    tab_profile, tab_model, tab_watch = st.sidebar.tabs(['Profile', 'Model', 'Watchlist'])

    # SIDEBAR TAB: Model (choose dataset + model)
    with tab_model:
        dataset = st.selectbox('Dataset', ['ml-1m', 'ml-32m'], key='dataset_choice')
        model_choice = st.selectbox('Model', ['LightGCN', 'BPR-MF'], key='model_choice')
        model_name = 'lightgcn' if model_choice == 'LightGCN' else 'bpr_mf'
        st.caption(f'Using: artifacts/{dataset}/{model_name}')

    # Load artifacts (once per rerun)
    U, I, items, train, val = load_artifacts(dataset, model_name)
    I = ensure_normalized_item_emb(I)
    index = build_ip_index(I)  # Index should store normalized copy; double-normalize is harmless

    # Genre preferences
    render_genre_preferences(items)

    # Year preferences
    render_year_preferences(items)

    # Ensure items index covers all embedding ids
    items.index = items.index.astype(int)
    n_items_from_emb = I.shape[0]
    missing = sorted(set(range(n_items_from_emb)) - set(items.index.astype(int).tolist()))
    if missing:
        st.warning(f'{len(missing)} item IDs missing in metadata; filling placeholders.')
        filler = pd.DataFrame(
            {
                'title': [f'Item {mid}' for mid in missing],  # per-id title
                'genres': [''] * len(missing),
                'year': [np.nan] * len(missing),
                '__pop__': [0] * len(missing),
            },
            index=pd.Index(missing, name=items.index.name or 'i'),
        )
        items = pd.concat([items, filler], axis=0).sort_index()

    # SIDEBAR TAB: Profile (profile settings)
    with tab_profile:
        st.subheader('Profile')
        st.session_state['user_id'] = st.text_input('User ID', value=st.session_state['user_id'])
        uploaded = st.file_uploader('Load profile (.json)', type=['json'])
        if uploaded is not None:
            try:
                st.session_state['profile'] = json.loads(uploaded.read().decode('utf-8'))
                st.success('Profile loaded')
            except Exception as e:
                st.error(f'Failed to load: {e}')

        # BUTTON: Download profile (as save not supported on Streamlit Community Cloud)
        payload = json.dumps(st.session_state['profile'], indent=2, ensure_ascii=False).encode('utf-8')
        if st.download_button(
            '💾 Download Profile',
            data=payload,
            file_name=f'{st.session_state["user_id"]}.json',
            mime='application/json',
            key='sidebar_download_profile',
        ):
            st.toast('Profile downloaded')

    # SIDEBAR TAB: Watchlist (items, watchlist boost)
    with tab_watch:
        render_watchlist_sidebar(items, key_prefix='global')

    # SIDEBAR TAB: Personalization controls last
    render_personal_knobs()

    # Main content tabs
    tab_recs, tab_setup = st.tabs(['🔮 Recommendations', '🧭 Profile Setup'])

    # MAIN TAB: Quick Profile Setup
    with tab_setup:
        st.subheader('Quick Profile Setup')
        prof = ensure_profile()

        #  Rapid review grid (popular/recent/random)
        st.markdown('#### Rapid Review')
        mode = st.radio('Pool', ['Popular','Recent','Random'], horizontal=True, key='setup_pool')
        page_size = st.selectbox('Cards per page', [12, 24, 48], index=1, key='setup_ps')

        # Build pool
        forbid = (
            set(map(int, prof.get('likes', []))) |
            set(map(int, prof.get('dislikes', []))) |
            set(map(int, prof.get('hidden', []))) |
            set(map(int, prof.get('watchlist', [])))
        )
        unseen = items.index.difference(pd.Index(sorted(forbid)))

        if mode == 'Popular':
            pop_series = train['i'].value_counts()
            order = pop_series.reindex(unseen).fillna(0).sort_values(ascending=False).index
        elif mode == 'Recent':
            order = items.loc[unseen].sort_values('year', ascending=False).index
        else:
            order = pd.Index(np.random.permutation(unseen))

        total_items = len(order)
        total_pages = max(1, (total_items + page_size - 1) // page_size)  # 1-based count

        # Show a 1-based page input
        display_page = st.number_input(
            'Page', min_value=1, max_value=total_pages,
            value=st.session_state.get('setup_page', 0) + 1,  # convert 0->1
            step=1, key='setup_page_display'
        )

        # Convert back to 0-based for indexing
        page = int(display_page) - 1
        st.session_state['setup_page'] = page

        # Clamp if feedback shrinks the pool
        max_page_idx = total_pages - 1
        if page > max_page_idx:
            st.session_state['setup_page'] = max_page_idx
            st.session_state['setup_page_display'] = max_page_idx + 1
            st.stop()

        # Slice the pool using 0-based page
        start = page * page_size
        pool = list(order[start : start + page_size])

        # Pagination with clamping (handles items disappearing after feedback)
        total = len(order)
        max_page = max(0, (total - 1) // page_size)
        if page > max_page:
            st.session_state['setup_page'] = max_page
            st.stop()  # re-run with clamped page

        start = page * page_size
        pool = list(order[start : start + page_size])

        # Render cards
        cols_per_row = 3
        for r in range(0, len(pool), cols_per_row):
            cols = st.columns(cols_per_row, gap='small')
            for c, it in enumerate(pool[r:r+cols_per_row]):
                with cols[c]:
                    title = re.sub(r'\s*\(\d{4}\)\s*$', '', str(items.loc[it,'title']))
                    year = items.loc[it,'year']
                    genres = items.loc[it,'genres']

                    st.markdown(f'**{title}** ({"" if pd.isna(year) else int(year)})  \n_{genres}_')

                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.button('👍', key=f'setup_like_{mode}_{page}_{int(it)}',
                                  help='Like - increases rank of similar titles',
                                on_click=apply_feedback, args=('like', int(it)))
                    with b2:
                        st.button('👎', key=f'setup_dislike_{mode}_{page}_{int(it)}',
                                  help='Dislike - downranks similar titles',
                                on_click=apply_feedback, args=('dislike', int(it)))
                    with b3:
                        st.button('🙈', key=f'setup_hide_{mode}_{page}_{int(it)}',
                                  help='Hide - remove from this list (non-punitive)',
                                on_click=apply_feedback, args=('hide', int(it)))
                    with b4:
                        starred = int(it) in st.session_state['profile'].get('watchlist', [])
                        label = '⭐' if not starred else '✅'
                        help_msg = 'Add to watchlist' if not starred else 'In watchlist (click to remove)'
                        st.button(label, key=f'setup_wl_{mode}_{page}_{int(it)}',
                                help=help_msg, on_click=watchlist_toggle, args=(int(it),))

        st.markdown('---')

        # BUTTON: Download profile (as save not supported on Streamlit Community Cloud)
        payload = json.dumps(st.session_state['profile'], indent=2, ensure_ascii=False).encode('utf-8')
        if st.download_button(
            '💾 Download Profile',
            data=payload,
            file_name=f'{st.session_state["user_id"]}.json',
            mime='application/json',
            key='setup_download_profile',
        ):
            st.toast('Profile downloaded')

        # BUTTON: Reset profile
        if st.button('🧹 Reset profile', key='setup_reset'):
            st.session_state['profile'] = {
                'likes': [], 'dislikes': [], 'hidden': [], 'watchlist': [],
                'genres': {}, 'year_pref': {'mu': 2000, 'sigma': 20},
                'weights': {'base': 1.0, 'sim': 0.7, 'genre': 0.2, 'year': 0.2, 'novel': 0.1, 'watch': 0.15, 'recency': 0.0},
                'ui': {'recency_half_life': 8, 'diversity_lambda': 0.85, 'watchlist_exclude': False},
            }            
            st.rerun()

    # MAIN TAB: Recommendations
    with tab_recs:
        st.subheader(f'top-k recommendations ({model_choice}, {dataset})')

        # Pull current profile (created/edited via sidebar or Profile Setup)
        prof = ensure_profile()
        learned_user = None  # # If we later authenticate users and have U[user_id_int]

        # Personal vector (Rocchio-style)
        ue = build_personal_vec(
            item_emb=I,
            likes=prof.get('likes', []),
            dislikes=prof.get('dislikes', []),
            learned_user=learned_user,
            alpha=0.7,
            beta=0.3,
            gamma=0.2,
        )
        if ue is None:
            st.info('No profile yet — open the **Profile Setup** tab or add a few likes on the left.')
            st.stop()

        K = st.slider('Top-K', min_value=10, max_value=50, value=20, step=5, key='topk_recs')

        forbid = (
            set(map(int, prof.get('likes', []))) |
            set(map(int, prof.get('hidden', []))) |
            set(map(int, prof.get('dislikes', []))) |
            set(map(int, prof.get('watchlist', [])))
        )
        cand = query_index(index, ue, topk=800, forbid=forbid)

        # Base score: cosine to personal vec (we can later blend model score)
        base_scores = I[cand] @ ue

        # Re-rank with soft prefs
        weights = prof.get('weights', {'base':1.0,'sim':0.7,'genre':0.2,'year':0.2,'novel':0.1,'watch':0.15})
        hl = float(prof.get('ui', {}).get('recency_half_life', 8))
        ranked, scores = rerank_candidates(
            user_vec=ue,
            cand_ids=cand,
            item_emb=I,
            base_scores=base_scores,
            items_df=items,
            genre_prefs=prof.get('genres', {}),  # Uses live genre sliders
            year_pref=prof.get('year_pref', {'mu': 2000, 'sigma': 20}),  # Uses live year sliders/decades
            watchlist_ids=set(prof.get('watchlist', [])),
            weights=weights,
            recency_half_life_years=hl,
        )
        if prof.get('ui', {}).get('watchlist_exclude', False):
            ranked = [r for r in ranked if r not in set(prof.get('watchlist', []))]

        # Apply MMR diversity using the lambda
        div_lambda = float(prof.get('ui', {}).get('diversity_lambda', 0.85))
        if div_lambda < 0.999:
            ranked = apply_mmr(ranked, scores, I, topk=K, lambda_div=div_lambda)
        else:
            ranked = ranked[:K]

        # Candidate generation (skipping already liked/hidden)
        topk = ranked[:K]

        for idx, it in enumerate(topk):
            title = items.loc[it, 'title']
            year = items.loc[it, 'year']
            genres = items.loc[it, 'genres']

            col1, col2 = st.columns([5, 1], vertical_alignment='center')
            with col1:
                # Recommended movie (year)
                st.markdown(f'**{idx+1}. {title}** ({"" if pd.isna(year) else int(year)})  \n_{genres}_')

                # Explain reasoning with a short, readable explanation
                exp = explain_recommendation(it, prof.get('likes', []), I, items, k_nn=2)
                if exp:
                    st.caption(exp)

            with col2:
                c1, c2, c3, c4 = st.columns(4)
                c1.button('👍', key=f'like_rec_{it}', help='Like',
                            on_click=apply_feedback, args=('like', int(it)))
                c2.button('👎', key=f'dislike_rec_{it}', help='Dislike',
                            on_click=apply_feedback, args=('dislike', int(it)))
                c3.button('🙈', key=f'hide_rec_{it}', help='Hide',
                            on_click=apply_feedback, args=('hide', int(it)))
                c4.button('⭐', key=f'wl_{it}', help='Add to watchlist',
                            on_click=watchlist_toggle, args=(int(it),))

        # Download profile (as save not supported on Streamlit Community Cloud)
        payload = json.dumps(st.session_state['profile'], indent=2, ensure_ascii=False).encode('utf-8')
        if st.download_button(
            '💾 Download Profile',
            data=payload,
            file_name=f'{st.session_state["user_id"]}.json',
            mime='application/json',
            key='download_profile_recs',
        ):
            st.toast('Profile downloaded')

        # Similar items explorer (based on liked list)
        st.markdown('---')
        st.subheader('🔎 Similar to a picked movie')
        liked_ids = list(map(int, prof.get('likes', [])))
        if liked_ids:
            choice = st.selectbox(
                'Choose one of your likes',
                liked_ids,
                format_func=lambda rid: str(items.loc[rid, 'title']),
                key=f'sim_from_like_{model_choice}'
            )
            vec = I[int(choice)]
            sims = I @ vec
            nn = np.argsort(-sims)[:K + 1]
            nn = [j for j in nn if j != int(choice) and j not in forbid][:K]
            similar_item = re.sub(r'\\s*\\(\\d{4}\\)\\s*$', '', str(items.loc[int(choice), 'title']))
            st.caption(f'Because you liked **{similar_item}**')
            for j in nn:
                title = items.loc[j, 'title']
                year = items.loc[j, 'year']
                genre = items.loc[j, 'genres']
                st.write(f'- {title} ({"" if pd.isna(year) else int(year)}) — _{genre}_')
        else:
            st.caption('Add at least one like to explore similar movies.')


if __name__ == '__main__':
    main()

