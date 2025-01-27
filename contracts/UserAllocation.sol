// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UserAllocation {
    struct Allocation {
        uint256 userId;
        uint256 bandwidth;
    }

    Allocation[] public allocations;
    event AllocationAdded(uint256 userId, uint256 bandwidth);

    function addAllocation(uint256 userId, uint256 bandwidth) public {
        allocations.push(Allocation(userId, bandwidth));
        emit AllocationAdded(userId, bandwidth);
    }

    function getAllocations() public view returns (Allocation[] memory) {
        return allocations;
    }
}
