# TEMPORARY FIX - Add this to your frontend auth.js temporarily
// In the login function, after the login fails, add:

// Temporary bypass for development
if (credentials.email === 'admin@supplychain.com' && credentials.password === 'admin123') {
    const fakeToken = 'fake-dev-token-' + Date.now();
    const fakeUser = {
        id: '550e8400-e29b-41d4-a716-446655440001',
        email: 'admin@supplychain.com',
        username: 'admin',
        role: 'admin',
        permissions: ['*']
    };
    
    localStorage.setItem('authToken', fakeToken);
    localStorage.setItem('user', JSON.stringify(fakeUser));
    api.defaults.headers.common['Authorization'] = `Bearer ${fakeToken}`;
    
    return {
        access_token: fakeToken,
        user: fakeUser
    };
}
