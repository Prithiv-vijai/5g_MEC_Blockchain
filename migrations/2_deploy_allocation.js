const UserAllocation = artifacts.require("UserAllocation");

module.exports = function (deployer) {
    deployer.deploy(UserAllocation);
};
